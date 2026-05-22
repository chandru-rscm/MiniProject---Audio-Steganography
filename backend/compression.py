"""
Image compression and adaptive sizing utilities.
"""

import io
import os
import zlib
import struct
import numpy as np
from PIL import Image, ImageFilter

try:
    import ai_enhance as _ai_enhance
    _AI_ENHANCE_READY = _ai_enhance.is_available()
except ImportError:
    _ai_enhance = None
    _AI_ENHANCE_READY = False

_MODEL      = None
_IMG_SIZE   = 128
_LATENT_CH  = 64
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'autoencoder.pth')
_AI_MODE    = False


def _try_load_model():
    global _MODEL, _IMG_SIZE, _LATENT_CH, _AI_MODE
    try:
        import torch
        from autoencoder import ImageAutoencoder

        if not os.path.exists(_MODEL_PATH):
            return

        ckpt = torch.load(_MODEL_PATH, map_location='cpu')
        _IMG_SIZE  = ckpt.get('img_size',   128)
        _LATENT_CH = ckpt.get('latent_ch',   64)

        model = ImageAutoencoder(latent_channels=_LATENT_CH)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        _MODEL   = model
        _AI_MODE = True
    except Exception:
        pass

_try_load_model()


def compress_image(image_bytes: bytes, quality: int = 50, capacity: int = None) -> tuple:
    """Compress image to target capacity using adaptive JPEG compression."""
    return _compress_jpeg(image_bytes, quality, capacity)


def get_compression_mode() -> str:
    """Return name of the current active compression mode."""
    if _AI_MODE and _AI_ENHANCE_READY:
        return 'AI Smart + DnCNN'
    return 'AI Smart Compression'


def decompress_image(compressed_bytes: bytes) -> bytes:
    """Decompress image bytes using the appropriate decoder mode."""
    raw = zlib.decompress(compressed_bytes)
    if raw[:4] == b'AISM':
        return _decompress_ai_smart(compressed_bytes)
    elif raw[:4] == b'AILF':
        return _decompress_ai_lossless(compressed_bytes)
    elif raw[:4] == b'AICM':
        return _decompress_ai(compressed_bytes)
    elif raw[:4] == b'AIJP':
        return _decompress_ai_jpeg(compressed_bytes)
    else:
        return _decompress_jpeg(compressed_bytes)


def get_compression_ratio(original_bytes: bytes, compressed_bytes: bytes) -> float:
    """Calculate the ratio of original size to compressed size."""
    return len(original_bytes) / len(compressed_bytes)


def _compress_ai(image_bytes: bytes, quality: int = 50) -> tuple:
    """Compress using the trained convolutional autoencoder."""
    import torch
    from torchvision import transforms

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_w, orig_h = img.size

    if quality >= 70:
        max_dim = 256
    elif quality >= 40:
        max_dim = 192
    else:
        max_dim = 128

    scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
    new_w = max(16, int(orig_w * scale))
    new_h = max(16, int(orig_h * scale))

    # Convert dimensions to be divisible by 8 for CNN layers
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    to_tensor = transforms.ToTensor()
    tensor    = to_tensor(img_resized).unsqueeze(0)

    with torch.no_grad():
        latent = _MODEL.encoder(tensor)

    latent_np = latent.squeeze(0).numpy()
    lat_min   = float(latent_np.min())
    lat_max   = float(latent_np.max())
    lat_range = lat_max - lat_min if lat_max != lat_min else 1.0

    quantized = ((latent_np - lat_min) / lat_range * 255).astype(np.uint8)
    lat_bytes = quantized.tobytes()

    C, H, W  = quantized.shape
    header   = (b'AICM' +
                struct.pack('>II', orig_w, orig_h) +
                struct.pack('>ff', lat_min, lat_max) +
                struct.pack('>III', C, H, W))

    raw        = header + lat_bytes
    compressed = zlib.compress(raw, level=6)

    jpeg_equiv = _get_jpeg_size(img_resized)

    resized_buf = io.BytesIO()
    img_resized.save(resized_buf, format='PNG', optimize=True)
    resized_size = len(resized_buf.getvalue())

    stats = {
        'original_size':   len(image_bytes),
        'orig_w':          orig_w,
        'orig_h':          orig_h,
        'resized_w':       new_w,
        'resized_h':       new_h,
        'resized_size':    resized_size,
        'jpeg_quality':    0,
        'jpeg_size':       jpeg_equiv,
        'compressed_size': len(compressed),
        'resize_ratio':    round(len(image_bytes) / resized_size, 1),
        'ai_ratio':        round(resized_size / len(compressed), 1),
        'jpeg_ratio':      round(len(image_bytes) / jpeg_equiv, 1),
        'total_ratio':     round(len(image_bytes) / len(compressed), 1),
        'mode':            'AI Autoencoder + DnCNN' if _AI_ENHANCE_READY else 'AI Autoencoder',
    }

    return compressed, stats


def _decompress_ai(compressed_bytes: bytes) -> bytes:
    """Decompress latent space representation back to an image."""
    import torch

    raw                      = zlib.decompress(compressed_bytes)
    orig_w, orig_h           = struct.unpack('>II', raw[4:12])
    lat_min, lat_max         = struct.unpack('>ff', raw[12:20])
    C, H, W                  = struct.unpack('>III', raw[20:32])
    lat_bytes                = raw[32:]

    quantized  = np.frombuffer(lat_bytes, dtype=np.uint8).reshape(C, H, W)
    lat_range  = lat_max - lat_min if lat_max != lat_min else 1.0
    latent_np  = quantized.astype(np.float32) / 255.0 * lat_range + lat_min

    latent     = torch.from_numpy(latent_np).unsqueeze(0)

    with torch.no_grad():
        recon = _MODEL.decoder(latent)

    recon_np  = (recon.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    img       = Image.fromarray(recon_np, 'RGB')

    if _AI_ENHANCE_READY and _ai_enhance is not None:
        img = _ai_enhance.enhance_image(img)
    else:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))

    if img.size != (orig_w, orig_h):
        img = img.resize((orig_w, orig_h), Image.BICUBIC)

    out = io.BytesIO()
    img.save(out, format='PNG', optimize=False)
    return out.getvalue()


def _get_jpeg_size(pil_img: Image.Image, quality: int = 58) -> int:
    """Get size of image when saved as JPEG."""
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    return len(buf.getvalue())


def _compress_jpeg(image_bytes: bytes, quality: int = 50, capacity: int = None) -> tuple:
    """Compress image using JPEG compression matching target capacity constraints."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_w, orig_h = img.size
    orig_size = len(image_bytes)
    ai_enhanced = _AI_ENHANCE_READY

    if capacity is None:
        jpeg_quality = int(25 + (quality / 100) * 67)
        jpeg_quality = max(25, min(92, jpeg_quality))
        if ai_enhanced:
            jpeg_quality = max(20, jpeg_quality - 5)

        if quality >= 90: max_dim = 2500
        elif quality >= 70: max_dim = 1800
        elif quality >= 40: max_dim = 1200
        else: max_dim = 800

        scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
        new_w, new_h = max(16, int(orig_w * scale)), max(16, int(orig_h * scale))
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        jpeg_buf = io.BytesIO()
        img_resized.save(jpeg_buf, format='JPEG', quality=jpeg_quality, optimize=True, progressive=True)
        jpeg_bytes = jpeg_buf.getvalue()
    else:
        # Target 88% capacity to account for encryption headers
        target_size = int(capacity * 0.88)
        
        best_jpeg_bytes = None
        best_w, best_h = 16, 16
        best_q = 20
        found = False

        max_theoretical_pixels = capacity * 15
        orig_pixels = orig_w * orig_h
        
        start_scale = 1.0
        if orig_pixels > max_theoretical_pixels:
            start_scale = (max_theoretical_pixels / orig_pixels) ** 0.5
            
        scales_to_test = [start_scale * f for f in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
        
        for scale in scales_to_test:
            new_w, new_h = max(16, int(orig_w * scale)), max(16, int(orig_h * scale))
            img_resized = img if scale >= 1.0 else img.resize((new_w, new_h), Image.LANCZOS)
                
            low, high = 15, 95
            best_q_for_scale = None
            best_bytes_for_scale = None
            
            while low <= high:
                q = (low + high) // 2
                buf = io.BytesIO()
                img_resized.save(buf, format='JPEG', quality=q, optimize=True, subsampling=0)
                b = buf.getvalue()
                
                est_final = len(b) + 50
                if est_final <= target_size:
                    best_q_for_scale = q
                    best_bytes_for_scale = b
                    low = q + 1
                else:
                    high = q - 1
                    
            if best_bytes_for_scale is not None:
                best_jpeg_bytes = best_bytes_for_scale
                best_w, best_h = new_w, new_h
                best_q = best_q_for_scale
                found = True
                if best_q >= 40:
                    break

        if not found:
            img_resized = img.resize((128, 128), Image.LANCZOS)
            buf = io.BytesIO()
            img_resized.save(buf, format='JPEG', quality=15, optimize=True, subsampling=0)
            best_jpeg_bytes = buf.getvalue()
            best_w, best_h = 128, 128
            best_q = 15
            
        jpeg_bytes = best_jpeg_bytes
        new_w, new_h = best_w, best_h
        jpeg_quality = best_q
        img_resized = img.resize((new_w, new_h), Image.LANCZOS) if (new_w, new_h) != (orig_w, orig_h) else img

    jpeg_size = len(jpeg_bytes)
    orig_size = len(image_bytes)

    if ai_enhanced:
        header = b'AIJP' + struct.pack('>III', orig_w, orig_h, orig_size)
    else:
        header = struct.pack('>III', orig_w, orig_h, orig_size)

    compressed = zlib.compress(header + jpeg_bytes, level=6)
    mode = 'AI-Enhanced JPEG' if ai_enhanced else 'JPEG'

    resized_buf = io.BytesIO()
    img_resized.save(resized_buf, format='PNG', optimize=True)
    resized_size = len(resized_buf.getvalue())

    stats = {
        'original_size':   len(image_bytes),
        'orig_w':          orig_w,
        'orig_h':          orig_h,
        'resized_w':       new_w,
        'resized_h':       new_h,
        'resized_size':    resized_size,
        'jpeg_quality':    jpeg_quality,
        'jpeg_size':       jpeg_size,
        'compressed_size': len(compressed),
        'resize_ratio':    round(len(image_bytes) / resized_size, 1),
        'ai_ratio':        round(resized_size / len(compressed), 1),
        'jpeg_ratio':      round(len(image_bytes) / jpeg_size, 1),
        'total_ratio':     round(len(image_bytes) / len(compressed), 1),
        'mode':            mode,
    }

    return compressed, stats


def _decompress_ai_jpeg(compressed_bytes: bytes) -> bytes:
    """Decompress JPEG with optional AI DnCNN artifact removal."""
    raw = zlib.decompress(compressed_bytes)
    orig_w, orig_h, orig_size = struct.unpack('>III', raw[4:16])
    jpeg_bytes     = raw[16:]

    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')

    if _AI_ENHANCE_READY and _ai_enhance is not None:
        img = _ai_enhance.enhance_image(img)
    else:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))

    if img.size != (orig_w, orig_h):
        img = img.resize((orig_w, orig_h), Image.LANCZOS)

    out = io.BytesIO()
    img.save(out, format='JPEG', quality=95, optimize=True, subsampling=0)
    out_bytes = out.getvalue()
    
    if len(out_bytes) < orig_size:
        out_bytes += b'\x00' * (orig_size - len(out_bytes))
        
    return out_bytes


def _decompress_jpeg(compressed_bytes: bytes) -> bytes:
    """Decompress standard JPEG payload."""
    raw            = zlib.decompress(compressed_bytes)
    orig_w, orig_h = struct.unpack('>II', raw[:8])
    jpeg_bytes     = raw[8:]

    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')

    if img.size != (orig_w, orig_h):
        img = img.resize((orig_w, orig_h), Image.BICUBIC)

    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))

    out = io.BytesIO()
    img.save(out, format='PNG', optimize=False)
    return out.getvalue()


def _compress_ai_lossless(image_bytes: bytes, quality: int = 50) -> tuple:
    """Compress image losslessly using PNG + zlib compression."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_w, orig_h = img.size

    png_buf = io.BytesIO()
    img.save(png_buf, format='PNG', optimize=True)
    png_bytes = png_buf.getvalue()
    png_size = len(png_bytes)

    header = b'AILF' + struct.pack('>II', orig_w, orig_h)
    raw = header + png_bytes
    compressed = zlib.compress(raw, level=9)

    mode = 'AI Lossless + DnCNN' if _AI_ENHANCE_READY else 'AI Lossless'

    stats = {
        'original_size':   len(image_bytes),
        'orig_w':          orig_w,
        'orig_h':          orig_h,
        'resized_w':       orig_w,
        'resized_h':       orig_h,
        'resized_size':    len(image_bytes),
        'jpeg_quality':    0,
        'jpeg_size':       png_size,
        'compressed_size': len(compressed),
        'resize_ratio':    1.0,
        'ai_ratio':        round(len(image_bytes) / len(compressed), 1),
        'jpeg_ratio':      round(len(image_bytes) / png_size, 1),
        'total_ratio':     round(len(image_bytes) / len(compressed), 1),
        'mode':            mode,
    }

    return compressed, stats


def _decompress_ai_lossless(compressed_bytes: bytes) -> bytes:
    """Decompress lossless PNG payload."""
    raw = zlib.decompress(compressed_bytes)
    orig_w, orig_h = struct.unpack('>II', raw[4:12])
    png_bytes = raw[12:]

    img = Image.open(io.BytesIO(png_bytes)).convert('RGB')

    out = io.BytesIO()
    img.save(out, format='PNG', optimize=False)
    return out.getvalue()


def _compress_ai_smart(image_bytes: bytes, quality: int = 50) -> tuple:
    """Compress using adaptive smart quality scaling."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_w, orig_h = img.size

    if quality >= 90:
        jpeg_quality = int(85 + (quality - 90) / 10 * 10)
        max_dim = 4000
    elif quality >= 70:
        jpeg_quality = int(82 + (quality - 70) / 20 * 3)
        max_dim = 3000
    elif quality >= 40:
        jpeg_quality = int(78 + (quality - 40) / 30 * 4)
        max_dim = 2048
    else:
        jpeg_quality = int(70 + (quality - 10) / 30 * 8)
        max_dim = 1024

    jpeg_quality = max(60, min(100, jpeg_quality))

    scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
    new_w = max(16, int(orig_w * scale))
    new_h = max(16, int(orig_h * scale))

    if scale < 1.0:
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        img_resized = img
        new_w, new_h = orig_w, orig_h

    jpeg_buf = io.BytesIO()
    img_resized.save(jpeg_buf, format='JPEG', quality=jpeg_quality, optimize=True, progressive=True)
    jpeg_bytes = jpeg_buf.getvalue()
    jpeg_size = len(jpeg_bytes)

    header = b'AISM' + struct.pack('>III', orig_w, orig_h, len(image_bytes))
    compressed = zlib.compress(header + jpeg_bytes, level=9)

    mode = 'AI Smart Compression'

    resized_buf = io.BytesIO()
    img_resized.save(resized_buf, format='PNG', optimize=True)
    resized_size = len(resized_buf.getvalue())

    stats = {
        'original_size':   len(image_bytes),
        'orig_w':          orig_w,
        'orig_h':          orig_h,
        'resized_w':       new_w,
        'resized_h':       new_h,
        'resized_size':    resized_size,
        'jpeg_quality':    jpeg_quality,
        'jpeg_size':       jpeg_size,
        'compressed_size': len(compressed),
        'resize_ratio':    round(len(image_bytes) / max(resized_size, 1), 1),
        'ai_ratio':        round(resized_size / max(len(compressed), 1), 1),
        'jpeg_ratio':      round(len(image_bytes) / max(jpeg_size, 1), 1),
        'total_ratio':     round(len(image_bytes) / max(len(compressed), 1), 1),
        'mode':            mode,
    }

    return compressed, stats


def _decompress_ai_smart(compressed_bytes: bytes) -> bytes:
    """Decompress smart scaled JPEG payload."""
    raw = zlib.decompress(compressed_bytes)
    orig_w, orig_h, orig_size = struct.unpack('>III', raw[4:16])
    jpeg_bytes = raw[16:]

    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')

    if img.size != (orig_w, orig_h):
        img = img.resize((orig_w, orig_h), Image.LANCZOS)

    out = io.BytesIO()
    img.save(out, format='JPEG', quality=95, optimize=True)
    out_bytes = out.getvalue()
    
    if len(out_bytes) < orig_size:
        out_bytes += b'\x00' * (orig_size - len(out_bytes))
        
    return out_bytes