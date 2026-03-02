"""
Audio Steganography using LSB (Least Significant Bit) technique.

KEY FIX: Work directly with raw int16 PCM using Python's built-in wave module.
The old version used soundfile floats, causing float<->int16 precision loss
that scrambled the LSBs on extraction.

Format: [MAGIC(4)] [DATA_LENGTH(8, big-endian uint64)] [DATA...]
"""

import io
import struct
import wave
import numpy as np


MAGIC = b'STEG'
BITS_PER_SAMPLE = 2
MASK = (1 << BITS_PER_SAMPLE) - 1   # 0b11 = 3
HEADER_SIZE = 4 + 8                  # MAGIC(4) + length(8)


# ── WAV I/O ───────────────────────────────────────────────────────────────────

def _read_wav_int16(audio_bytes: bytes):
    """
    Read audio and return (int16 numpy array, wave params).
    Converts non-PCM formats to PCM_16 WAV first via soundfile.
    """
    buf = io.BytesIO(audio_bytes)
    try:
        with wave.open(buf) as wf:
            p = wf.getparams()
            if p.sampwidth == 2:  # already int16 PCM
                raw = wf.readframes(p.nframes)
                samples = np.frombuffer(raw, dtype=np.int16).copy()
                return samples, p
    except Exception:
        pass

    # Fallback: use soundfile to decode, then re-encode as PCM_16 WAV
    import soundfile as sf
    buf.seek(0)
    data, sr = sf.read(buf, always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]  # take first channel if stereo
    samples = np.clip(np.round(data * 32767), -32768, 32767).astype(np.int16)

    # Write as proper PCM_16 WAV
    out = io.BytesIO()
    with wave.open(out, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    out.seek(0)
    with wave.open(out) as wf:
        p = wf.getparams()
    return samples, p


def _write_wav_int16(samples: np.ndarray, params) -> bytes:
    """Write int16 samples as PCM_16 WAV with original params."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


# ── Public API ────────────────────────────────────────────────────────────────

def get_capacity(audio_bytes: bytes) -> int:
    """Return max bytes that can be hidden in this audio."""
    samples, _ = _read_wav_int16(audio_bytes)
    return (len(samples) * BITS_PER_SAMPLE) // 8 - HEADER_SIZE


def hide_data(audio_bytes: bytes, secret: bytes) -> bytes:
    """Hide secret bytes in audio LSBs. Raises ValueError if too small."""
    samples, params = _read_wav_int16(audio_bytes)

    payload = MAGIC + struct.pack('>Q', len(secret)) + secret
    capacity = (len(samples) * BITS_PER_SAMPLE) // 8

    if len(payload) > capacity:
        raise ValueError(
            f"Audio too small: capacity {capacity} bytes, "
            f"need {len(payload)} bytes. "
            f"Use a longer audio file or lower the quality setting."
        )

    chunks = _bytes_to_chunks(payload)

    # Embed directly into int16 — no float conversion!
    s = samples.astype(np.int32)
    for i, chunk in enumerate(chunks):
        s[i] = (s[i] & ~MASK) | int(chunk)
    samples = s.astype(np.int16)

    return _write_wav_int16(samples, params)


def extract_data(stego_audio_bytes: bytes) -> bytes:
    """Extract hidden data from stego audio. Raises ValueError if not found."""
    samples, _ = _read_wav_int16(stego_audio_bytes)
    s = samples.astype(np.int32)

    # Read header
    header_n = HEADER_SIZE * (8 // BITS_PER_SAMPLE)  # number of samples for header
    header_chunks = [int(s[i]) & MASK for i in range(header_n)]
    header_bytes = _chunks_to_bytes(header_chunks)

    if header_bytes[:4] != MAGIC:
        raise ValueError("No valid hidden data found in this audio file.")

    data_length = struct.unpack('>Q', header_bytes[4:12])[0]

    if data_length == 0 or data_length > 50 * 1024 * 1024:
        raise ValueError("No valid hidden data found in this audio file.")

    data_n = data_length * (8 // BITS_PER_SAMPLE)
    if header_n + data_n > len(s):
        raise ValueError("Audio too short: data may be corrupted.")

    data_chunks = [int(s[i]) & MASK for i in range(header_n, header_n + data_n)]
    return _chunks_to_bytes(data_chunks)


# ── Bit helpers ───────────────────────────────────────────────────────────────

def _bytes_to_chunks(data: bytes) -> list:
    """Split bytes into 2-bit chunks, MSB first. Each byte → 4 chunks."""
    chunks = []
    for byte in data:
        chunks.append((byte >> 6) & 0x03)
        chunks.append((byte >> 4) & 0x03)
        chunks.append((byte >> 2) & 0x03)
        chunks.append((byte >> 0) & 0x03)
    return chunks


def _chunks_to_bytes(chunks: list) -> bytes:
    """Combine 2-bit chunks into bytes, 4 chunks per byte, MSB first."""
    result = []
    for i in range(0, len(chunks) - 3, 4):
        byte = (
            (chunks[i]   & 0x03) << 6 |
            (chunks[i+1] & 0x03) << 4 |
            (chunks[i+2] & 0x03) << 2 |
            (chunks[i+3] & 0x03)
        )
        result.append(byte)
    return bytes(result)
