"""
AI Image Compression using JPEG + zlib pipeline.

Compression pipeline:
  1. Resize to max 384-640px depending on quality
  2. JPEG encode at mapped quality (25-92)
  3. zlib on JPEG bytes for extra 10-20% reduction

Decompression:
  1. zlib decompress
  2. JPEG decode
  3. Bicubic upsample to original size
  4. Unsharp mask for crispness
"""

import io
import zlib
import struct
from PIL import Image, ImageFilter


def compress_image(image_bytes: bytes, quality: int = 50) -> tuple:
    """
    Compress image using JPEG + zlib.

    quality: 1-100
      - 80-90: Very good, larger payload (~300-500KB for 4MB image)
      - 50-70: Good quality, medium payload (~100-250KB)
      - 20-40: Acceptable, small payload (~40-100KB)
      - 10-20: Low quality, tiny payload (~20-50KB)

    Returns:
        tuple: (compressed_bytes, stats_dict)
            compressed_bytes — zlib(header + JPEG), ready for encryption
            stats_dict — all intermediate sizes for UI display
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    # Map quality slider (1-100) to internal JPEG quality (25-92)
    jpeg_quality = int(25 + (quality / 100) * 67)
    jpeg_quality = max(25, min(92, jpeg_quality))

    # Determine resize target based on quality
    if quality >= 70:
        max_dim = 640
    elif quality >= 40:
        max_dim = 512
    else:
        max_dim = 384

    # Resize preserving aspect ratio
    scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
    new_w = max(16, int(orig_w * scale))
    new_h = max(16, int(orig_h * scale))

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # JPEG encode — this is the viewable image format size
    jpeg_buf = io.BytesIO()
    img_resized.save(jpeg_buf, format='JPEG', quality=jpeg_quality,
                     optimize=True, progressive=True)
    jpeg_bytes = jpeg_buf.getvalue()
    jpeg_size = len(jpeg_bytes)  # ← actual image format size (viewable JPEG)

    # Pack header: orig_w, orig_h
    header = struct.pack('>II', orig_w, orig_h)

    # zlib compress the header + JPEG bytes
    compressed = zlib.compress(header + jpeg_bytes, level=6)
    compressed_size = len(compressed)

    # Build stats dict for UI
    stats = {
        'original_size':     len(image_bytes),          # raw input bytes
        'orig_w':            orig_w,
        'orig_h':            orig_h,
        'resized_w':         new_w,
        'resized_h':         new_h,
        'jpeg_quality':      jpeg_quality,
        'jpeg_size':         jpeg_size,                  # viewable JPEG size
        'compressed_size':   compressed_size,            # after zlib (what gets encrypted)
        'jpeg_ratio':        round(len(image_bytes) / jpeg_size, 1),
        'total_ratio':       round(len(image_bytes) / compressed_size, 1),
    }

    print(f"[COMPRESS] {len(image_bytes)/1024:.0f}KB "
          f"→ resize {orig_w}x{orig_h}→{new_w}x{new_h} "
          f"→ JPEG q{jpeg_quality}: {jpeg_size/1024:.1f}KB "
          f"→ zlib: {compressed_size/1024:.1f}KB "
          f"(ratio {stats['total_ratio']}x)")

    return compressed, stats


def decompress_image(compressed_bytes: bytes) -> bytes:
    """
    Decompress image back to original dimensions.
    Returns PNG bytes.
    """
    raw = zlib.decompress(compressed_bytes)
    orig_w, orig_h = struct.unpack('>II', raw[:8])
    jpeg_bytes = raw[8:]

    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

    if img.size != (orig_w, orig_h):
        img = img.resize((orig_w, orig_h), Image.BICUBIC)

    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))

    out = io.BytesIO()
    img.save(out, format='PNG', optimize=False)
    return out.getvalue()


def get_compression_ratio(original_bytes: bytes, compressed_bytes: bytes) -> float:
    return len(original_bytes) / len(compressed_bytes)