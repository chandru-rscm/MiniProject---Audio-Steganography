"""
Quality Metrics — StegoWave
Optimized with NumPy to handle massive 100+ Megapixel images instantly.
"""

import io
import wave
import math
import numpy as np
from PIL import Image

# ══════════════════════════════════════════════════════════════════
#  AUDIO METRICS
# ══════════════════════════════════════════════════════════════════

def compute_audio_metrics(original_bytes: bytes, stego_bytes: bytes) -> dict:
    buf_o = io.BytesIO(original_bytes)
    with wave.open(buf_o) as wf:
        orig = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        
    buf_s = io.BytesIO(stego_bytes)
    with wave.open(buf_s) as wf:
        stego = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    n = min(len(orig), len(stego))
    orig = orig[:n].astype(np.float64)
    stego = stego[:n].astype(np.float64)

    noise = stego - orig
    mse = np.mean(noise**2)
    signal_power = np.mean(orig**2)
    
    snr = float('inf') if mse == 0 else 10 * np.log10(signal_power / mse)
    psnr = float('inf') if mse == 0 else 10 * np.log10((32767.0**2) / mse)
    
    # Correlation
    mean_o = np.mean(orig)
    mean_s = np.mean(stego)
    den = np.sqrt(np.sum((orig - mean_o)**2) * np.sum((stego - mean_s)**2))
    correlation = 1.0 if den == 0 else np.sum((orig - mean_o) * (stego - mean_s)) / den
    
    modified = np.sum(noise != 0)
    pct_mod = (modified / n) * 100

    # BER for audio
    orig_bits = np.unpackbits(orig.astype(np.int16).view(np.uint8))
    stego_bits = np.unpackbits(stego.astype(np.int16).view(np.uint8))
    ber = np.sum(orig_bits != stego_bits) / len(orig_bits)

    return {
        'snr': float(round(snr, 2)),
        'psnr': float(round(psnr, 2)),
        'mse': float(round(mse, 4)),
        'correlation': float(round(correlation, 6)),
        'pct_modified': float(round(pct_mod, 4)),
        'ber': float(round(ber, 6)),
        'total_samples': int(n),
        'modified_samples': int(modified),
    }

# ══════════════════════════════════════════════════════════════════
#  IMAGE METRICS
# ══════════════════════════════════════════════════════════════════

def compute_image_metrics(original_bytes: bytes, recovered_bytes: bytes) -> dict:
    orig_img = Image.open(io.BytesIO(original_bytes)).convert('RGB')
    rec_img = Image.open(io.BytesIO(recovered_bytes)).convert('RGB')
    
    if rec_img.size != orig_img.size:
        rec_img = rec_img.resize(orig_img.size, Image.LANCZOS)
        
    orig = np.array(orig_img, dtype=np.float64)
    rec = np.array(rec_img, dtype=np.float64)
    
    mse = np.mean((orig - rec) ** 2)
    rmse = np.sqrt(mse)
    psnr = float('inf') if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
    
    # SSIM (Global)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(orig, axis=(0,1))
    mu2 = np.mean(rec, axis=(0,1))
    var1 = np.var(orig, axis=(0,1))
    var2 = np.var(rec, axis=(0,1))
    cov12 = np.mean((orig - mu1) * (rec - mu2), axis=(0,1))
    
    ssim_channels = ((2 * mu1 * mu2 + C1) * (2 * cov12 + C2)) / ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))
    ssim_val = np.mean(ssim_channels)
    
    # BER
    orig_bits = np.unpackbits(orig.astype(np.uint8))
    rec_bits = np.unpackbits(rec.astype(np.uint8))
    ber = np.sum(orig_bits != rec_bits) / len(orig_bits)

    return {
        'psnr': float(round(psnr, 2)),
        'ssim': float(round(ssim_val, 4)),
        'ssim_pct': float(round(ssim_val * 100, 2)),
        'mse': float(round(mse, 2)),
        'rmse': float(round(rmse, 2)),
        'ber': float(round(ber, 6))
    }