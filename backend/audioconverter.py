"""
Audio format converter utility supporting MP3 and WAV conversions.
"""

import io
import wave
import array
import os

# Dynamic PATH addition for local FFmpeg binaries on Windows
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + _backend_dir


def is_mp3(audio_bytes: bytes) -> bool:
    """Check if the provided audio file bytes represent an MP3 format."""
    if len(audio_bytes) < 3:
        return False
    if audio_bytes[:3] == b'ID3':
        return True
    if audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0:
        return True
    return False


def is_wav(audio_bytes: bytes) -> bool:
    """Check if the provided audio file bytes represent a WAV format."""
    return audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE'


def mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Convert MP3 audio bytes to raw PCM WAV bytes."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for MP3 support. "
            "Install with: pip install pydub\n"
            "Also install ffmpeg: https://ffmpeg.org/download.html"
        )

    mp3_buf = io.BytesIO(mp3_bytes)
    audio   = AudioSegment.from_mp3(mp3_buf)

    # Convert to mono 16-bit PCM (required for LSB embedding)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)

    wav_buf = io.BytesIO()
    audio.export(wav_buf, format='wav')
    wav_buf.seek(0)
    return wav_buf.read()


def wav_to_mp3(wav_bytes: bytes, bitrate: str = '192k') -> bytes:
    """Convert WAV audio bytes to MP3 format."""
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
    """Ensure audio is in WAV format, converting from MP3 if necessary."""
    if is_wav(audio_bytes):
        return audio_bytes, 'wav'
    elif is_mp3(audio_bytes):
        print("[AUDIO] MP3 detected → converting to WAV for LSB processing...")
        wav_bytes = mp3_to_wav(audio_bytes)
        print(f"[AUDIO] Converted: {len(audio_bytes)/1024:.1f}KB MP3 "
              f"→ {len(wav_bytes)/1024:.1f}KB WAV")
        return wav_bytes, 'mp3'
    else:
        return audio_bytes, 'wav'


def get_audio_info(audio_bytes: bytes) -> dict:
    """Fetch basic metadata and properties of the audio file."""
    if is_wav(audio_bytes):
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf) as wf:
            params = wf.getparams()
        return {
            'format':      'WAV',
            'channels':    params.nchannels,
            'sample_rate': params.framerate,
            'sample_width': params.sampwidth * 8,
            'n_frames':    params.nframes,
            'duration_s':  round(params.nframes / params.framerate, 2),
        }
    elif is_mp3(audio_bytes):
        return {'format': 'MP3', 'note': 'Will be converted to WAV for processing'}
    return {'format': 'Unknown'}