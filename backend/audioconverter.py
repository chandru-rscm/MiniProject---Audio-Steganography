"""
Audio Format Converter — StegoWave

Handles MP3 ↔ WAV conversion so the LSB steganography pipeline
(which works only on WAV/PCM) can accept MP3 carrier files.

Workflow for MP3 input:
    1. Detect if input is MP3
    2. Convert MP3 → WAV PCM using pydub
    3. Run LSB embed/extract on WAV (existing pipeline unchanged)
    4. Convert WAV → MP3 back for output

Why MP3 needs special handling:
    MP3 is a lossy compressed format — it does NOT store raw PCM samples.
    LSB embedding requires raw int16 PCM samples to modify directly.
    So we must decode MP3 → raw PCM (WAV) first, do the embedding,
    then re-encode back to MP3.

    IMPORTANT: MP3 re-encoding is lossy — it will destroy the embedded LSBs!
    So the output stego file must stay as WAV if the receiver needs to decrypt.
    MP3 output is only for the carrier preview — not for the stego file itself.

    For actual steganography:
        - Carrier can be MP3 (we convert it to WAV internally)
        - Output stego file is ALWAYS WAV (to preserve LSBs)
        - Receiver must use the WAV stego file for decryption

Requires:
    pip install pydub
    ffmpeg installed on system (pydub uses it for MP3 decode)

    On Windows: download ffmpeg from https://ffmpeg.org/download.html
                add to PATH or place ffmpeg.exe in backend folder
"""

import io
import wave
import array
import os

# Dynamic PATH addition for local FFmpeg binaries on Windows
# This ensures that if ffmpeg.exe and ffprobe.exe are present in the backend folder,
# they are automatically found by pydub without needing global installation.
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + _backend_dir


def is_mp3(audio_bytes: bytes) -> bool:
    """
    Detect if audio bytes are MP3 format.
    MP3 files start with:
        - ID3 tag: b'ID3'
        - MPEG sync: 0xFF 0xFB / 0xFF 0xFA / 0xFF 0xF3 etc.
    """
    if len(audio_bytes) < 3:
        return False
    # ID3 tag header
    if audio_bytes[:3] == b'ID3':
        return True
    # MPEG sync bytes
    if audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0:
        return True
    return False


def is_wav(audio_bytes: bytes) -> bool:
    """WAV files start with RIFF header."""
    return audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE'


def mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """
    Convert MP3 bytes to WAV PCM bytes using pydub.

    Returns WAV bytes ready for LSB embedding.
    Converts to mono 16-bit PCM at original sample rate.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for MP3 support. "
            "Install with: pip install pydub\n"
            "Also install ffmpeg: https://ffmpeg.org/download.html"
        )

    # Load MP3
    mp3_buf = io.BytesIO(mp3_bytes)
    audio   = AudioSegment.from_mp3(mp3_buf)

    # Convert to mono 16-bit PCM (required for LSB embedding)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)   # 2 bytes = 16-bit

    # Export as WAV
    wav_buf = io.BytesIO()
    audio.export(wav_buf, format='wav')
    wav_buf.seek(0)
    return wav_buf.read()


def wav_to_mp3(wav_bytes: bytes, bitrate: str = '192k') -> bytes:
    """
    Convert WAV bytes back to MP3 using pydub.

    NOTE: This re-encoding is lossy and will DESTROY any LSB data!
    Only use this for the carrier audio output (not the stego audio).
    The stego audio must always remain WAV.

    bitrate: '128k', '192k', '320k'
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub required. pip install pydub")

    wav_buf = io.BytesIO(wav_bytes)
    audio   = AudioSegment.from_wav(wav_buf)

    mp3_buf = io.BytesIO()
    audio.export(mp3_buf, format='mp3', bitrate=bitrate)
    mp3_buf.seek(0)
    return mp3_buf.read()


def ensure_wav(audio_bytes: bytes) -> tuple:
    """
    Ensure audio is in WAV format for LSB processing.

    Returns:
        (wav_bytes, original_format)
        original_format: 'wav' or 'mp3'

    If input is already WAV → return as-is.
    If input is MP3 → convert to WAV and return.
    """
    if is_wav(audio_bytes):
        return audio_bytes, 'wav'
    elif is_mp3(audio_bytes):
        print("[AUDIO] MP3 detected → converting to WAV for LSB processing...")
        wav_bytes = mp3_to_wav(audio_bytes)
        print(f"[AUDIO] Converted: {len(audio_bytes)/1024:.1f}KB MP3 "
              f"→ {len(wav_bytes)/1024:.1f}KB WAV")
        return wav_bytes, 'mp3'
    else:
        # Unknown format — try as WAV anyway
        return audio_bytes, 'wav'


def get_audio_info(audio_bytes: bytes) -> dict:
    """
    Get basic info about audio file.
    Works for WAV. For MP3, returns format only.
    """
    if is_wav(audio_bytes):
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf) as wf:
            params = wf.getparams()
        return {
            'format':      'WAV',
            'channels':    params.nchannels,
            'sample_rate': params.framerate,
            'sample_width': params.sampwidth * 8,  # in bits
            'n_frames':    params.nframes,
            'duration_s':  round(params.nframes / params.framerate, 2),
        }
    elif is_mp3(audio_bytes):
        return {'format': 'MP3', 'note': 'Will be converted to WAV for processing'}
    return {'format': 'Unknown'}