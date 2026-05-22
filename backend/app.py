"""
Flask API for Audio Steganography.
"""

import os
import io
import zlib
import base64
import traceback
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from compression import compress_image, decompress_image, get_compression_ratio, get_compression_mode
from steganography import hide_data, extract_data, get_capacity
from crypto import encrypt, decrypt
from metrics import compute_audio_metrics, compute_image_metrics

app = Flask(__name__)
CORS(app)

_last_preview_jpeg  = None
_last_hist_original = None
_last_hist_stego    = None

MAX_FILE_SIZE = 50 * 1024 * 1024


@app.route('/')
def index():
    """Serve the index page from the frontend directory."""
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
    return send_from_directory(frontend_dir, 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'message': 'Audio Steganography API running'})


@app.route('/api/capacity', methods=['POST'])
def check_capacity():
    """Check storage capacity of the provided audio file."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    try:
        capacity = get_capacity(audio_bytes)
        return jsonify({
            'capacity_bytes': capacity,
            'capacity_kb':    round(capacity / 1024, 2),
            'capacity_mb':    round(capacity / (1024 * 1024), 4),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/encrypt', methods=['POST'])
def encrypt_endpoint():
    """Compress, encrypt and hide an image inside an audio file."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    if not request.form.get('password'):
        return jsonify({'error': 'Password is required'}), 400

    image_file = request.files['image']
    audio_file = request.files['audio']
    password   = request.form.get('password')
    quality    = int(request.form.get('quality', 50))
    quality    = max(1, min(100, quality))

    try:
        image_bytes = image_file.read()
        audio_bytes = audio_file.read()

        capacity       = get_capacity(audio_bytes)
        audio_size     = len(audio_bytes)
        
        compressed, comp_stats = compress_image(image_bytes, quality=quality, capacity=capacity)
        compressed_size = len(compressed)

        global _last_preview_jpeg
        raw = zlib.decompress(compressed)
        if raw[:4] == b'AISM':
            _last_preview_jpeg = raw[16:]
        elif raw[:4] == b'AILF':
            _last_preview_jpeg = raw[12:]
        elif raw[:4] == b'AICM':
            _last_preview_jpeg = decompress_image(compressed)
        elif raw[:4] == b'AIJP':
            _last_preview_jpeg = raw[16:]
        else:
            _last_preview_jpeg = raw[8:]

        encrypted      = encrypt(compressed, password)
        encrypted_size = len(encrypted)

        total_samples  = capacity * 4
        lsb_bits       = 2
        lsb_possib     = 4
        samples_used   = encrypted_size * 4

        if encrypted_size > capacity:
            return jsonify({
                'error': (
                    f'Audio file too small. '
                    f'Capacity: {capacity/1024:.1f} KB, '
                    f'Required: {encrypted_size/1024:.1f} KB. '
                    f'Try a lower quality setting or a longer audio file.'
                )
            }), 400

        stego_audio = hide_data(audio_bytes, encrypted)
        audio_metrics = compute_audio_metrics(audio_bytes, stego_audio)

        global _last_hist_original, _last_hist_stego
        _last_hist_original = _compute_histogram(audio_bytes)
        _last_hist_stego    = _compute_histogram(stego_audio)

        out = io.BytesIO(stego_audio)
        out.seek(0)

        response = send_file(
            out,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='stego_audio.wav'
        )

        h = response.headers
        h['X-Original-Size']     = str(comp_stats['original_size'])
        h['X-Original-W']        = str(comp_stats['orig_w'])
        h['X-Original-H']        = str(comp_stats['orig_h'])
        h['X-Resized-W']         = str(comp_stats['resized_w'])
        h['X-Resized-H']         = str(comp_stats['resized_h'])
        h['X-JPEG-Quality']      = str(comp_stats['jpeg_quality'])
        h['X-JPEG-Size']         = str(comp_stats['jpeg_size'])
        h['X-Resized-Size']      = str(comp_stats.get('resized_size', 0))
        h['X-Resize-Ratio']      = str(comp_stats.get('resize_ratio', 0))
        h['X-AI-Ratio']          = str(comp_stats.get('ai_ratio', 0))
        h['X-Compressed-Size']   = str(comp_stats['compressed_size'])
        h['X-JPEG-Ratio']        = str(comp_stats['jpeg_ratio'])
        h['X-Total-Ratio']       = str(comp_stats['total_ratio'])
        h['X-Encrypted-Size']    = str(encrypted_size)
        h['X-Compression-Mode']  = get_compression_mode()

        h['X-Audio-Size']        = str(audio_size)
        h['X-Audio-Capacity']    = str(capacity)
        h['X-Total-Samples']     = str(total_samples)
        h['X-Samples-Used']      = str(samples_used)
        h['X-LSB-Bits']          = str(lsb_bits)
        h['X-LSB-Possibilities'] = str(lsb_possib)

        h['X-Audio-SNR']         = str(audio_metrics['snr'])
        h['X-Audio-PSNR']        = str(audio_metrics['psnr'])
        h['X-Audio-MSE']         = str(audio_metrics['mse'])
        h['X-Audio-Correlation'] = str(audio_metrics['correlation'])
        h['X-Audio-Pct-Modified']= str(audio_metrics['pct_modified'])

        h['Access-Control-Expose-Headers'] = (
            'X-Original-Size, X-Original-W, X-Original-H, '
            'X-Resized-W, X-Resized-H, X-JPEG-Quality, '
            'X-JPEG-Size, X-Resized-Size, X-Resize-Ratio, X-AI-Ratio, '
            'X-Compressed-Size, X-JPEG-Ratio, '
            'X-Total-Ratio, X-Encrypted-Size, X-Compression-Mode, '
            'X-Audio-Size, X-Audio-Capacity, '
            'X-Total-Samples, X-Samples-Used, '
            'X-LSB-Bits, X-LSB-Possibilities, '
            'X-Audio-SNR, X-Audio-PSNR, X-Audio-MSE, '
            'X-Audio-Correlation, X-Audio-Pct-Modified'
        )
        return response

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/api/decrypt', methods=['POST'])
def decrypt_endpoint():
    """Extract, decrypt and reconstruct image from stego audio."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    if not request.form.get('password'):
        return jsonify({'error': 'Password is required'}), 400

    audio_file    = request.files['audio']
    password      = request.form.get('password')
    original_file = request.files.get('original_image')

    try:
        stego_bytes = audio_file.read()

        encrypted = extract_data(stego_bytes)
        compressed = decrypt(encrypted, password)
        image_bytes = decompress_image(compressed)

        out = io.BytesIO(image_bytes)
        out.seek(0)
        
        extension = 'png'
        mime_type = 'image/png'
        if image_bytes.startswith(b'\xff\xd8'):
            extension = 'jpg'
            mime_type = 'image/jpeg'
            
        response = send_file(
            out,
            mimetype=mime_type,
            as_attachment=True,
            download_name=f'recovered_image.{extension}'
        )

        if original_file:
            try:
                orig_bytes   = original_file.read()
                img_metrics  = compute_image_metrics(orig_bytes, image_bytes)
                h = response.headers
                h['X-Image-PSNR']     = str(img_metrics['psnr'])
                h['X-Image-SSIM']     = str(img_metrics['ssim'])
                h['X-Image-SSIM-Pct'] = str(img_metrics['ssim_pct'])
                h['X-Image-MSE']      = str(img_metrics['mse'])
                h['X-Image-RMSE']     = str(img_metrics['rmse'])
                h['Access-Control-Expose-Headers'] = (
                    'X-Image-PSNR, X-Image-SSIM, '
                    'X-Image-SSIM-Pct, X-Image-MSE, X-Image-RMSE'
                )
            except Exception as me:
                print(f"[DECRYPT] Metrics error (non-fatal): {me}")

        return response

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


def _compute_histogram(audio_bytes: bytes, bins: int = 128) -> list:
    """Compute amplitude histogram of audio samples."""
    import wave
    import array as arr

    buf = io.BytesIO(audio_bytes)
    with wave.open(buf) as wf:
        raw = wf.readframes(wf.getnframes())

    samples   = arr.array('h', raw)
    bin_size  = 65536 // bins
    counts    = [0] * bins

    for s in samples:
        idx = (s + 32768) // bin_size
        idx = min(idx, bins - 1)
        counts[idx] += 1

    return counts


@app.route('/api/histogram', methods=['GET'])
def get_histogram():
    """Fetch amplitude histogram of original and stego audio files."""
    if _last_hist_original is None or _last_hist_stego is None:
        return jsonify({'error': 'No histogram data. Run encrypt first.'}), 404

    return jsonify({
        'original': _last_hist_original,
        'stego':    _last_hist_stego,
        'bins':     len(_last_hist_original),
    })


@app.route('/api/preview-compressed', methods=['GET'])
def preview_compressed():
    """Fetch compressed preview of the hidden payload image."""
    global _last_preview_jpeg
    if _last_preview_jpeg is None:
        return jsonify({'error': 'No preview available. Run encrypt first.'}), 404
    out = io.BytesIO(_last_preview_jpeg)
    out.seek(0)
    
    is_png = _last_preview_jpeg.startswith(b'\x89PNG')
    
    return send_file(
        out,
        mimetype='image/png' if is_png else 'image/jpeg',
        as_attachment=False,
        download_name='compressed_preview.png' if is_png else 'compressed_preview.jpg'
    )


if __name__ == '__main__':
    print("=" * 60)
    print("  Audio Steganography Server")
    print("  Running on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)