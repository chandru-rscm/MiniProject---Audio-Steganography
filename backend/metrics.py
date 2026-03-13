"""
Quality metrics for StegoWave.

Audio metrics  (original WAV vs stego WAV):
  - SNR   : Signal-to-Noise Ratio (dB)  — how inaudible the change is
  - PSNR  : Peak SNR (dB)               — standard audio quality measure
  - MSE   : Mean Square Error           — average sample-level change
  - Correlation : Pearson correlation   — waveform similarity (0-1)

Image metrics  (original image vs recovered image):
  - PSNR  : Peak Signal-to-Noise Ratio (dB) — reconstruction quality
  - SSIM  : Structural Similarity Index (0-1) — perceptual similarity
  - MSE   : Mean Square Error                 — pixel-level error
  - RMSE  : Root MSE                          — in pixel units (0-255)
"""

import io
import wave
import struct
import math
import numpy as np
from PIL import Image


# ══════════════════════════════════════════════════════════════════
#  AUDIO METRICS
# ══════════════════════════════════════════════════════════════════

def _read_wav_samples(wav_bytes: bytes) -> np.ndarray:
    """Read raw int16 PCM samples from WAV bytes."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float64)


def compute_audio_metrics(original_bytes: bytes, stego_bytes: bytes) -> dict:
    """
    Compare original WAV vs stego WAV and return quality metrics.

    Returns dict with SNR, PSNR, MSE, correlation — all rounded.
    """
    orig  = _read_wav_samples(original_bytes)
    stego = _read_wav_samples(stego_bytes)

    # Trim to same length (stego should be identical length, just in case)
    n = min(len(orig), len(stego))
    orig  = orig[:n]
    stego = stego[:n]

    noise = stego - orig   # difference = LSB noise introduced

    # MSE
    mse = float(np.mean(noise ** 2))

    # SNR = 10 * log10(signal_power / noise_power)
    signal_power = float(np.mean(orig ** 2))
    noise_power  = float(np.mean(noise ** 2))
    if noise_power == 0:
        snr = float('inf')
    else:
        snr = 10 * math.log10(signal_power / noise_power)

    # PSNR = 10 * log10(MAX² / MSE)  where MAX = 32767 for int16
    MAX_VAL = 32767.0
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10((MAX_VAL ** 2) / mse)

    # Pearson correlation coefficient
    if np.std(orig) == 0 or np.std(stego) == 0:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(orig, stego)[0, 1])

    # Percentage of samples that were actually modified
    modified_samples = int(np.sum(noise != 0))
    total_samples    = n
    pct_modified     = round((modified_samples / total_samples) * 100, 4)

    return {
        'snr':              round(snr, 2),
        'psnr':             round(psnr, 2),
        'mse':              round(mse, 4),
        'correlation':      round(correlation, 6),
        'pct_modified':     pct_modified,
        'total_samples':    total_samples,
        'modified_samples': modified_samples,
    }


# ══════════════════════════════════════════════════════════════════
#  IMAGE METRICS
# ══════════════════════════════════════════════════════════════════

def _img_to_array(image_bytes: bytes) -> np.ndarray:
    """Load image bytes → float64 numpy array, RGB, range 0-255."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return np.array(img, dtype=np.float64)


def compute_image_metrics(original_bytes: bytes, recovered_bytes: bytes) -> dict:
    """
    Compare original image vs recovered image and return quality metrics.

    Returns PSNR, SSIM, MSE, RMSE — all rounded.
    """
    orig_arr = _img_to_array(original_bytes)
    recv_arr = _img_to_array(recovered_bytes)

    # Resize recovered to match original dimensions if different
    if orig_arr.shape != recv_arr.shape:
        orig_img = Image.open(io.BytesIO(original_bytes)).convert('RGB')
        recv_img = Image.open(io.BytesIO(recovered_bytes)).convert('RGB')
        recv_img = recv_img.resize(orig_img.size, Image.BICUBIC)
        recv_arr = np.array(recv_img, dtype=np.float64)

    diff = orig_arr - recv_arr

    # MSE across all pixels and channels
    mse  = float(np.mean(diff ** 2))
    rmse = math.sqrt(mse)

    # PSNR
    MAX_VAL = 255.0
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10((MAX_VAL ** 2) / mse)

    # SSIM (per channel, then average)
    ssim_val = _ssim(orig_arr, recv_arr)

    return {
        'psnr':       round(psnr, 2),
        'ssim':       round(ssim_val, 4),
        'ssim_pct':   round(ssim_val * 100, 2),   # e.g. 82.5%
        'mse':        round(mse, 2),
        'rmse':       round(rmse, 2),
    }


def _ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute mean SSIM across all channels.
    Uses standard constants: C1=(0.01*255)^2, C2=(0.03*255)^2
    Window-free version (global statistics) — fast and good enough for demo.
    """
    C1 = (0.01 * 255) ** 2   # 6.5025
    C2 = (0.03 * 255) ** 2   # 58.5225

    ssim_channels = []
    for c in range(img1.shape[2]):   # R, G, B
        ch1 = img1[:, :, c]
        ch2 = img2[:, :, c]

        mu1    = np.mean(ch1)
        mu2    = np.mean(ch2)
        sigma1 = np.std(ch1)
        sigma2 = np.std(ch2)
        sigma12 = np.mean((ch1 - mu1) * (ch2 - mu2))

        numerator   = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)

        ssim_channels.append(numerator / denominator)

    return float(np.mean(ssim_channels))