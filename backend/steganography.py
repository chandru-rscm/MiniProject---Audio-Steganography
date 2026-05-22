"""
Audio LSB Steganography library.
"""

import io
import struct
import wave
import array

MAGIC           = b'STEG'
BITS_PER_SAMPLE = 2
MASK            = (1 << BITS_PER_SAMPLE) - 1
CHUNKS_PER_BYTE = 8 // BITS_PER_SAMPLE
HEADER_SIZE     = 4 + 8


def _read_wav_samples(audio_bytes: bytes):
    """Read WAV file and return samples array and wave parameters."""
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf) as wf:
        params = wf.getparams()
        raw    = wf.readframes(params.nframes)

    samples = array.array('h', raw)
    return samples, params


def _write_wav_samples(samples: array.array, params) -> bytes:
    """Write samples back to PCM WAV format."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def get_capacity(audio_bytes: bytes) -> int:
    """Return maximum payload capacity of the WAV audio in bytes."""
    samples, _ = _read_wav_samples(audio_bytes)
    total_storable = (len(samples) * BITS_PER_SAMPLE) // 8
    return total_storable - HEADER_SIZE


def hide_data(audio_bytes: bytes, secret: bytes) -> bytes:
    """Hide secret data in the audio's LSBs."""
    samples, params = _read_wav_samples(audio_bytes)

    # Payload = MAGIC + length(8 bytes) + secret data
    payload  = MAGIC + struct.pack('>Q', len(secret)) + secret

    capacity = (len(samples) * BITS_PER_SAMPLE) // 8
    if len(payload) > capacity:
        raise ValueError(
            f"Audio too small: capacity {capacity/1024:.1f} KB, "
            f"need {len(payload)/1024:.1f} KB."
        )

    chunks = _bytes_to_chunks(payload)
    inv_mask = ~MASK

    for i in range(len(chunks)):
        sample = samples[i]
        sample_int = sample if sample >= 0 else sample + 65536
        modified = (sample_int & inv_mask) | chunks[i]
        if modified > 32767:
            modified -= 65536
        samples[i] = modified

    return _write_wav_samples(samples, params)


def extract_data(stego_audio_bytes: bytes) -> bytes:
    """Extract hidden data from stego audio file."""
    samples, _ = _read_wav_samples(stego_audio_bytes)

    header_chunk_count = HEADER_SIZE * CHUNKS_PER_BYTE
    if len(samples) < header_chunk_count:
        raise ValueError("Audio file too short to contain hidden data.")

    header_chunks = []
    for i in range(header_chunk_count):
        sample = samples[i]
        chunk  = sample & MASK
        header_chunks.append(chunk)

    header_bytes = _chunks_to_bytes(header_chunks)

    if header_bytes[:4] != MAGIC:
        raise ValueError("No valid hidden data found in this audio file.")

    secret_length = struct.unpack('>Q', header_bytes[4:12])[0]
    if secret_length == 0 or secret_length > 150 * 1024 * 1024:
        raise ValueError("Invalid data length.")

    total_chunks_need = header_chunk_count + (secret_length * CHUNKS_PER_BYTE)
    if total_chunks_need > len(samples):
        raise ValueError("Audio too short: hidden data appears corrupted.")

    data_chunks = []
    for i in range(header_chunk_count, total_chunks_need):
        sample = samples[i]
        chunk  = sample & MASK
        data_chunks.append(chunk)

    return _chunks_to_bytes(data_chunks)


def _bytes_to_chunks(data: bytes) -> list:
    """Split bytes into 2-bit chunks (MSB first)."""
    chunks = []
    for byte in data:
        chunks.append((byte >> 6) & 0x03)
        chunks.append((byte >> 4) & 0x03)
        chunks.append((byte >> 2) & 0x03)
        chunks.append((byte >> 0) & 0x03)
    return chunks


def _chunks_to_bytes(chunks: list) -> bytes:
    """Reassemble 2-bit chunks back into bytes (MSB first)."""
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