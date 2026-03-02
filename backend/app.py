"""
Flask backend for Audio Steganography with AI Compression.

Endpoints:
  POST /api/encrypt  - Hide image in audio
  POST /api/decrypt  - Extract image from stego audio
  POST /api/capacity - Check audio capacity
"""

import os
import io
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from compression import compress_image, decompress_image, get_compression_ratio
from steganography import hide_data, extract_data, get_capacity
from crypto import encrypt, decrypt

app = Flask(__name__)
CORS(app)

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
            'capacity_kb': round(capacity / 1024, 2),
            'capacity_mb': round(capacity / (1024 * 1024), 4),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/encrypt', methods=['POST'])
def encrypt_endpoint():
    """
    Encrypt: Compress image → AES encrypt → LSB hide in audio → return stego audio.
    
    Form data:
      - image: image file
      - audio: audio file  
      - password: string
      - quality: int 1-100 (optional, default 50)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    if not request.form.get('password'):
        return jsonify({'error': 'Password is required'}), 400

    image_file = request.files['image']
    audio_file = request.files['audio']
    password = request.form.get('password')
    quality = int(request.form.get('quality', 50))
    quality = max(1, min(100, quality))

    try:
        image_bytes = image_file.read()
        audio_bytes = audio_file.read()
        orig_image_size = len(image_bytes)

        # Step 1: AI Compress image
        print(f"[ENCRYPT] Compressing image ({orig_image_size/1024:.1f} KB) with quality={quality}...")
        compressed = compress_image(image_bytes, quality=quality)
        compressed_size = len(compressed)
        ratio = get_compression_ratio(image_bytes, compressed)
        print(f"[ENCRYPT] Compressed: {compressed_size/1024:.1f} KB (ratio: {ratio:.1f}x)")

        # Step 2: AES-256-GCM encrypt
        print("[ENCRYPT] Encrypting...")
        encrypted = encrypt(compressed, password)
        encrypted_size = len(encrypted)

        # Step 3: Check audio capacity
        capacity = get_capacity(audio_bytes)
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

        # Step 4: LSB steganography
        print("[ENCRYPT] Hiding data in audio...")
        stego_audio = hide_data(audio_bytes, encrypted)
        print(f"[ENCRYPT] Done. Stego audio size: {len(stego_audio)/1024:.1f} KB")

        # Return stego audio as WAV
        out = io.BytesIO(stego_audio)
        out.seek(0)

        response = send_file(
            out,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='stego_audio.wav'
        )
        # Add stats headers
        response.headers['X-Original-Image-Size'] = str(orig_image_size)
        response.headers['X-Compressed-Size'] = str(compressed_size)
        response.headers['X-Compression-Ratio'] = str(round(ratio, 2))
        response.headers['X-Encrypted-Size'] = str(encrypted_size)
        response.headers['Access-Control-Expose-Headers'] = (
            'X-Original-Image-Size, X-Compressed-Size, '
            'X-Compression-Ratio, X-Encrypted-Size'
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
      - audio: stego audio file
      - password: string
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    if not request.form.get('password'):
        return jsonify({'error': 'Password is required'}), 400

    audio_file = request.files['audio']
    password = request.form.get('password')

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
        return send_file(
            out,
            mimetype='image/png',
            as_attachment=True,
            download_name='recovered_image.png'
        )

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  Audio Steganography API")
    print("  Running on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
