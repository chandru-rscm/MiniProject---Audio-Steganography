"""
PSNR Verification Script - StegoWave

Tests the full pipeline: compress -> encrypt -> hide -> extract -> decrypt -> decompress
Then measures PSNR, SSIM, MSE between input and output images.

Usage:
    python verify_psnr.py                    # test with generated images
    python verify_psnr.py path/to/image.jpg  # test specific image
    python verify_psnr.py path/to/folder/    # batch test all images in folder
"""

import os
import sys
import io
import math
import glob
from PIL import Image

# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))

from compression import compress_image, decompress_image
from crypto import encrypt, decrypt
from steganography import hide_data, extract_data
from metrics import compute_image_metrics


def create_test_image(width=256, height=256, name="gradient"):
    """Create a simple test image for verification."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            r = int((x / width) * 255)
            g = int((y / height) * 255)
            b = int(((x + y) / (width + height)) * 255)
            pixels[x, y] = (r, g, b)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue(), name


def test_compression_only(image_bytes, label="test"):
    """Test compress → decompress (no steganography)."""
    print(f"\n{'_' * 60}")
    print(f"  Testing: {label}")
    print(f"  Input size: {len(image_bytes)/1024:.1f} KB")
    print(f"{'_' * 60}")

    # Compress
    compressed, stats = compress_image(image_bytes, quality=50)
    print(f"  Mode: {stats['mode']}")
    print(f"  Compressed: {len(compressed)/1024:.1f} KB (ratio: {stats['total_ratio']}x)")

    # Decompress
    recovered = decompress_image(compressed)
    print(f"  Recovered: {len(recovered)/1024:.1f} KB")

    # Metrics
    metrics = compute_image_metrics(image_bytes, recovered)
    print(f"\n  +-- Image Quality Metrics --------------------+")
    print(f"  |  PSNR:  {metrics['psnr']:>10} dB {'OK' if metrics['psnr'] >= 40 or metrics['psnr'] == float('inf') else 'FAIL'}                  |")
    print(f"  |  SSIM:  {metrics['ssim']:>10}    {'OK' if metrics['ssim'] >= 0.99 else 'WARN'}                  |")
    print(f"  |  MSE:   {metrics['mse']:>10}                          |")
    print(f"  |  RMSE:  {metrics['rmse']:>10}                          |")
    print(f"  +------------------------------------------------+")

    return metrics


def test_full_pipeline(image_bytes, audio_bytes, password, label="test"):
    """Test full pipeline: compress -> encrypt -> hide -> extract -> decrypt -> decompress."""
    print(f"{'=' * 60}")
    print(f"  FULL PIPELINE TEST: {label}")
    print(f"  Image: {len(image_bytes)/1024:.1f} KB | Audio: {len(audio_bytes)/1024:.1f} KB")
    print(f"{'=' * 60}")

    # Step 1: Compress
    compressed, stats = compress_image(image_bytes, quality=50)
    print(f"  [1] Compressed: {len(compressed)/1024:.1f} KB ({stats['mode']})")

    # Step 2: Encrypt
    encrypted = encrypt(compressed, password)
    print(f"  [2] Encrypted:  {len(encrypted)/1024:.1f} KB")

    # Step 3: Hide in audio
    try:
        stego = hide_data(audio_bytes, encrypted)
        print(f"  [3] Hidden in audio: {len(stego)/1024:.1f} KB")
    except ValueError as e:
        print(f"  [3] [X] Audio too small: {e}")
        print(f"      Need audio with at least {len(encrypted)*4} samples")
        print(f"      Falling back to compression-only test...")
        return test_compression_only(image_bytes, label)

    # Step 4: Extract
    extracted = extract_data(stego)
    print(f"  [4] Extracted:  {len(extracted)/1024:.1f} KB")

    # Step 5: Decrypt
    decrypted = decrypt(extracted, password)
    print(f"  [5] Decrypted:  {len(decrypted)/1024:.1f} KB")

    # Step 6: Decompress
    recovered = decompress_image(decrypted)
    print(f"  [6] Recovered:  {len(recovered)/1024:.1f} KB")

    # Verify data integrity
    assert compressed == decrypted, "[X] Data integrity check failed!"
    print(f"  [OK] Data integrity verified (compress == decrypt)")

    # Metrics
    metrics = compute_image_metrics(image_bytes, recovered)
    psnr_ok = metrics['psnr'] >= 40 or metrics['psnr'] == float('inf')
    ssim_ok = metrics['ssim'] >= 0.99

    print(f"\n  +-- Image Quality Metrics --------------------+")
    print(f"  |  PSNR:  {metrics['psnr']:>10} dB {'OK' if psnr_ok else 'FAIL'}                  |")
    print(f"  |  SSIM:  {metrics['ssim']:>10}    {'OK' if ssim_ok else 'WARN'}                  |")
    print(f"  |  MSE:   {metrics['mse']:>10}                          |")
    print(f"  |  RMSE:  {metrics['rmse']:>10}                          |")
    print(f"  +------------------------------------------------+")

    return metrics


def generate_test_audio(duration_seconds=30, sample_rate=44100):
    """Generate a silent WAV file for testing."""
    import wave
    import array

    n_samples = sample_rate * duration_seconds
    samples = array.array('h', [0] * n_samples)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return buf.getvalue()


def main():
    print("=" * 60)
    print("  [PSNR] StegoWave PSNR Verification")
    print("  Testing image quality through the full pipeline")
    print("=" * 60)

    password = "test_password_123"
    results = []

    # Generate test audio (30 seconds mono WAV = ~2.6MB, capacity ~650KB)
    print("\n  Generating test audio (30s mono WAV)...")
    audio = generate_test_audio(duration_seconds=30)
    print(f"  Audio size: {len(audio)/1024:.1f} KB")

    if len(sys.argv) > 1:
        # Test specific images from args
        targets = sys.argv[1:]
        for target in targets:
            if os.path.isdir(target):
                # Batch mode
                patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
                files = []
                for p in patterns:
                    files.extend(glob.glob(os.path.join(target, p)))
                for f in sorted(files):
                    with open(f, 'rb') as fp:
                        img_bytes = fp.read()
                    label = os.path.basename(f)
                    m = test_full_pipeline(img_bytes, audio, password, label)
                    results.append((label, m))
            elif os.path.isfile(target):
                with open(target, 'rb') as fp:
                    img_bytes = fp.read()
                label = os.path.basename(target)
                m = test_full_pipeline(img_bytes, audio, password, label)
                results.append((label, m))
            else:
                print(f"  [!] Not found: {target}")
    else:
        # Test with generated images
        for size in [64, 128, 256]:
            img_bytes, name = create_test_image(size, size, f"gradient_{size}x{size}")
            m = test_full_pipeline(img_bytes, audio, password, name)
            results.append((name, m))

    # Summary table
    if results:
        print(f"\n\n{'=' * 60}")
        print(f"  [SUMMARY]")
        print(f"{'=' * 60}")
        print(f"  {'Image':<30} {'PSNR (dB)':>12} {'SSIM':>10} {'Result':>8}")
        print(f"  {'_' * 62}")
        all_pass = True
        for label, m in results:
            psnr_str = f"{m['psnr']}" if m['psnr'] != float('inf') else "INF"
            ok = m['psnr'] >= 40 or m['psnr'] == float('inf')
            if not ok:
                all_pass = False
            print(f"  {label:<30} {psnr_str:>12} {m['ssim']:>10.4f} {'PASS' if ok else 'FAIL':>8}")

        print(f"\n  {'[OK] ALL TESTS PASSED - PSNR >= 40 dB' if all_pass else '[FAIL] SOME TESTS FAILED'}")
        print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
