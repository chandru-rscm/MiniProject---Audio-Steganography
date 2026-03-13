"""
Flask backend for Audio Steganography with AI Compression.

Endpoints:
  POST /api/encrypt  - Hide image in audio
  POST /api/decrypt  - Extract image from stego audio
  POST /api/capacity - Check audio capacity
"""

import os
import io
import zlib
import base64
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from compression import compress_image, decompress_image, get_compression_ratio
from steganography import hide_data, extract_data, get_capacity
from crypto import encrypt, decrypt
from metrics import compute_audio_metrics, compute_image_metrics

app = Flask(__name__)
CORS(app)

# Temp store for compressed image preview (single-user demo app)
_last_preview_jpeg = None

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Audio Steganography API running'})


@app.route('/api/capacity', methods=['POST'])
def check_capacity():
    """Check how many bytes an audio file can hide."""
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
    """
    Encrypt: Compress image → AES encrypt → LSB hide in audio → return stego audio.

    Form data:
      - image:    image file
      - audio:    audio file
      - password: string
      - quality:  int 1-100 (optional, default 50)
    """
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

        # ── Step 1: AI Compress image ────────────────────────────────
        print(f"[ENCRYPT] Compressing image ({len(image_bytes)/1024:.1f} KB) quality={quality}...")
        compressed, comp_stats = compress_image(image_bytes, quality=quality)
        compressed_size = len(compressed)

        # Extract JPEG bytes for preview (zlib decompress → skip 8-byte header)
        global _last_preview_jpeg
        raw = zlib.decompress(compressed)
        _last_preview_jpeg = raw[8:]   # pure JPEG bytes, viewable image

        # ── Step 2: AES-256-GCM encrypt ──────────────────────────────
        print("[ENCRYPT] Encrypting...")
        encrypted      = encrypt(compressed, password)
        encrypted_size = len(encrypted)

        # ── Step 3: Check audio capacity ─────────────────────────────
        capacity       = get_capacity(audio_bytes)
        audio_size     = len(audio_bytes)
        total_samples  = capacity * 4          # each sample holds 2 bits = 0.25 bytes → samples = capacity*4
        lsb_bits       = 2
        lsb_possib     = 4                     # 2^2 = 4 possibilities per sample: 00,01,10,11
        samples_used   = encrypted_size * 4    # each byte needs 4 samples

        print(f"[ENCRYPT] Audio capacity: {capacity/1024:.1f} KB, need: {encrypted_size/1024:.1f} KB")

        if encrypted_size > capacity:
            return jsonify({
                'error': (
                    f'Audio file too small. '
                    f'Capacity: {capacity/1024:.1f} KB, '
                    f'Required: {encrypted_size/1024:.1f} KB. '
                    f'Try a lower quality setting or a longer audio file.'
                )
            }), 400

        # ── Step 4: LSB steganography ─────────────────────────────────
        print("[ENCRYPT] Hiding data in audio...")
        stego_audio = hide_data(audio_bytes, encrypted)
        print(f"[ENCRYPT] Done. Stego audio size: {len(stego_audio)/1024:.1f} KB")

        # ── Step 5: Audio quality metrics ────────────────────────────
        print("[ENCRYPT] Computing audio metrics...")
        audio_metrics = compute_audio_metrics(audio_bytes, stego_audio)

        # ── Return stego audio ────────────────────────────────────────
        out = io.BytesIO(stego_audio)
        out.seek(0)

        response = send_file(
            out,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='stego_audio.wav'
        )

        # ── All stats in headers ──────────────────────────────────────
        h = response.headers

        # Image stats
        h['X-Original-Size']     = str(comp_stats['original_size'])
        h['X-Original-W']        = str(comp_stats['orig_w'])
        h['X-Original-H']        = str(comp_stats['orig_h'])
        h['X-Resized-W']         = str(comp_stats['resized_w'])
        h['X-Resized-H']         = str(comp_stats['resized_h'])
        h['X-JPEG-Quality']      = str(comp_stats['jpeg_quality'])
        h['X-JPEG-Size']         = str(comp_stats['jpeg_size'])       # viewable image size
        h['X-Compressed-Size']   = str(comp_stats['compressed_size']) # after zlib
        h['X-JPEG-Ratio']        = str(comp_stats['jpeg_ratio'])
        h['X-Total-Ratio']       = str(comp_stats['total_ratio'])
        h['X-Encrypted-Size']    = str(encrypted_size)

        # Audio stats
        h['X-Audio-Size']        = str(audio_size)
        h['X-Audio-Capacity']    = str(capacity)
        h['X-Total-Samples']     = str(total_samples)
        h['X-Samples-Used']      = str(samples_used)
        h['X-LSB-Bits']          = str(lsb_bits)
        h['X-LSB-Possibilities'] = str(lsb_possib)

        # Audio quality metrics
        h['X-Audio-SNR']         = str(audio_metrics['snr'])
        h['X-Audio-PSNR']        = str(audio_metrics['psnr'])
        h['X-Audio-MSE']         = str(audio_metrics['mse'])
        h['X-Audio-Correlation'] = str(audio_metrics['correlation'])
        h['X-Audio-Pct-Modified']= str(audio_metrics['pct_modified'])

        h['Access-Control-Expose-Headers'] = (
            'X-Original-Size, X-Original-W, X-Original-H, '
            'X-Resized-W, X-Resized-H, X-JPEG-Quality, '
            'X-JPEG-Size, X-Compressed-Size, X-JPEG-Ratio, '
            'X-Total-Ratio, X-Encrypted-Size, '
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
    """
    Decrypt: Extract from stego audio → AES decrypt → Decompress → return image.

    Form data:
      - audio:          stego audio file
      - password:       string
      - original_image: (optional) original image for PSNR/SSIM comparison
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    if not request.form.get('password'):
        return jsonify({'error': 'Password is required'}), 400

    audio_file    = request.files['audio']
    password      = request.form.get('password')
    original_file = request.files.get('original_image')   # optional

    try:
        stego_bytes = audio_file.read()

        # Step 1: LSB extract
        print("[DECRYPT] Extracting hidden data from audio...")
        encrypted = extract_data(stego_bytes)
        print(f"[DECRYPT] Extracted {len(encrypted)/1024:.1f} KB")

        # Step 2: AES decrypt
        print("[DECRYPT] Decrypting...")
        compressed = decrypt(encrypted, password)
        print(f"[DECRYPT] Decrypted: {len(compressed)/1024:.1f} KB")

        # Step 3: AI Decompress
        print("[DECRYPT] Decompressing (AI reconstruction)...")
        image_bytes = decompress_image(compressed)
        print(f"[DECRYPT] Reconstructed image: {len(image_bytes)/1024:.1f} KB")

        out = io.BytesIO(image_bytes)
        out.seek(0)
        response = send_file(
            out,
            mimetype='image/png',
            as_attachment=True,
            download_name='recovered_image.png'
        )

        # ── Image quality metrics (only if original provided) ─────────
        if original_file:
            try:
                print("[DECRYPT] Computing image quality metrics...")
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
                print(f"[DECRYPT] PSNR={img_metrics['psnr']} dB  SSIM={img_metrics['ssim_pct']}%")
            except Exception as me:
                print(f"[DECRYPT] Metrics error (non-fatal): {me}")

        return response

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/api/preview-compressed', methods=['GET'])
def preview_compressed():
    """Return the last compressed JPEG as a downloadable image."""
    global _last_preview_jpeg
    if _last_preview_jpeg is None:
        return jsonify({'error': 'No preview available. Run encrypt first.'}), 404
    out = io.BytesIO(_last_preview_jpeg)
    out.seek(0)
    return send_file(
        out,
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='compressed_preview.jpg'
    )


if __name__ == '__main__':
    print("=" * 60)
    print("  Audio Steganography API")
    print("  Running on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)