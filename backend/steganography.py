"""
Audio Steganography using LSB (Least Significant Bit) technique.

Implemented from scratch using ONLY Python built-in modules:
    wave   — read/write WAV files
    struct — pack/unpack binary data
    array  — typed array of int16 samples (no numpy)

NO external libraries used in the core algorithm.

How LSB embedding works:
    - Each audio sample is a 16-bit signed integer (range -32768 to 32767)
    - We take the last 2 bits (LSBs) of each sample to store our data
    - 2 bits per sample = 4 possible values: 00, 01, 10, 11
    - Each byte of secret data needs 4 samples (4 x 2bits = 8bits = 1 byte)
    - Change in sample value is at most 3 out of 32767 = 0.009% — inaudible

Bit operations used:
    MASK        = 0b11 = 3         (isolates last 2 bits)
    sample & ~MASK                 (clears last 2 bits)
    (sample & ~MASK) | chunk       (inserts 2-bit chunk into cleared bits)
    sample & MASK                  (extracts last 2 bits)

Payload format stored in audio:
    [MAGIC: 4 bytes 'STEG'] [LENGTH: 8 bytes big-endian uint64] [DATA: N bytes]

Each byte of payload:
    MSB → [bit7 bit6] [bit5 bit4] [bit3 bit2] [bit1 bit0] ← LSB
             chunk0      chunk1      chunk2      chunk3
    stored in 4 consecutive audio samples
"""

import io
import struct
import wave
import array


MAGIC           = b'STEG'          # 4-byte magic header to verify extraction
BITS_PER_SAMPLE = 2                # Strict 2-bit LSB for academic panel compliance
MASK            = (1 << BITS_PER_SAMPLE) - 1   # 0b11 = 3
CHUNKS_PER_BYTE = 8 // BITS_PER_SAMPLE         # 4 chunks make 1 byte
HEADER_SIZE     = 4 + 8            # MAGIC(4 bytes) + data length(8 bytes)


# ── WAV I/O ───────────────────────────────────────────────────────────────────

def _read_wav_samples(audio_bytes: bytes):
    """
    Read WAV file and return (samples_array, wave_params).

    samples_array: Python array.array of signed int16 samples
    wave_params:   namedtuple from wave module (nchannels, sampwidth, etc.)

    Uses only Python built-in wave module — no numpy, no soundfile.
    """
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf) as wf:
        params = wf.getparams()
        raw    = wf.readframes(params.nframes)

    # array module gives us a typed array of int16 — no numpy needed
    # 'h' type code = signed short = 16-bit integer
    samples = array.array('h', raw)
    return samples, params


def _write_wav_samples(samples: array.array, params) -> bytes:
    """
    Write int16 samples back as PCM WAV.
    Uses only Python built-in wave module.
    """
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# ── Public API ────────────────────────────────────────────────────────────────

def get_capacity(audio_bytes: bytes) -> int:
    """
    Return max number of bytes that can be hidden in this audio.

    Formula:
        total_samples * BITS_PER_SAMPLE / 8  =  total bytes storable
        subtract HEADER_SIZE for the MAGIC + length prefix
    """
    samples, _ = _read_wav_samples(audio_bytes)
    total_storable = (len(samples) * BITS_PER_SAMPLE) // 8
    return total_storable - HEADER_SIZE


def hide_data(audio_bytes: bytes, secret: bytes) -> bytes:
    """
    Hide secret bytes inside audio using PURE LSB steganography (4-bit).
    No EOF appending tricks. Strict academic LSB.
    """
    samples, params = _read_wav_samples(audio_bytes)

    # Build full payload with header
    payload  = MAGIC + struct.pack('>Q', len(secret)) + secret

    # Calculate physical LSB capacity
    capacity = (len(samples) * BITS_PER_SAMPLE) // 8
    if len(payload) > capacity:
        raise ValueError(
            f"Audio too small: capacity {capacity/1024:.1f} KB, "
            f"need {len(payload)/1024:.1f} KB. "
            f"Use a longer audio file or lower the quality setting."
        )

    # Convert LSB payload bytes to 2-bit chunks
    chunks = _bytes_to_chunks(payload)

    # Embed each chunk into one audio sample
    inv_mask = ~MASK  # precompute once

    for i in range(len(chunks)):
        sample = samples[i]
        sample_int = sample if sample >= 0 else sample + 65536
        modified = (sample_int & inv_mask) | chunks[i]
        if modified > 32767:
            modified -= 65536
        samples[i] = modified

    return _write_wav_samples(samples, params)


def extract_data(stego_audio_bytes: bytes) -> bytes:
    """
    Extract hidden bytes from stego audio using strict LSB reading.
    """
    samples, _ = _read_wav_samples(stego_audio_bytes)

    # Read header first from LSB
    header_chunk_count = HEADER_SIZE * CHUNKS_PER_BYTE

    if len(samples) < header_chunk_count:
        raise ValueError("Audio file too short to contain hidden data.")

    header_chunks = []
    for i in range(header_chunk_count):
        sample = samples[i]
        chunk  = sample & MASK
        header_chunks.append(chunk)

    header_bytes = _chunks_to_bytes(header_chunks)

    # Verify MAGIC
    if header_bytes[:4] != MAGIC:
        raise ValueError("No valid hidden data found in this audio file.")

    # Read total secret data length (8-byte big-endian unsigned int)
    secret_length = struct.unpack('>Q', header_bytes[4:12])[0]

    if secret_length == 0 or secret_length > 150 * 1024 * 1024:
        raise ValueError("Invalid data length — audio may not contain hidden data.")

    total_chunks_need = header_chunk_count + (secret_length * CHUNKS_PER_BYTE)

    if total_chunks_need > len(samples):
        raise ValueError("Audio too short: hidden data appears corrupted.")

    data_chunks = []
    for i in range(header_chunk_count, total_chunks_need):
        sample = samples[i]
        chunk  = sample & MASK
        data_chunks.append(chunk)

    return _chunks_to_bytes(data_chunks)


# ── Bit helpers ───────────────────────────────────────────────────────────────

def _bytes_to_chunks(data: bytes) -> list:
    """
    Split each byte into 4 x 2-bit chunks, MSB first.
    """
    chunks = []
    for byte in data:
        chunks.append((byte >> 6) & 0x03)   # bits 7-6
        chunks.append((byte >> 4) & 0x03)   # bits 5-4
        chunks.append((byte >> 2) & 0x03)   # bits 3-2
        chunks.append((byte >> 0) & 0x03)   # bits 1-0
    return chunks


def _chunks_to_bytes(chunks: list) -> bytes:
    """
    Reassemble 4 x 2-bit chunks back into 1 byte, MSB first.
    """
    result = []
    for i in range(0, len(chunks) - (CHUNKS_PER_BYTE - 1), CHUNKS_PER_BYTE):
        byte = (
            (chunks[i]   & 0x03) << 6 |
            (chunks[i+1] & 0x03) << 4 |
            (chunks[i+2] & 0x03) << 2 |
            (chunks[i+3] & 0x03) << 0
        )
        result.append(byte)
    return bytes(result)