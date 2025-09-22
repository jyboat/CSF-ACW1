"""
I/O:
- Input: path to a WAV file (RIFF/WAVE), expected PCM integer format (8/16/24-bit), mono/stereo.
- Output: CoverMetaAudio (metadata) and capacity-in-bytes functions.

Spec mapping:
- Capacity requirement before encoding, and limit check with error if payload too large.
"""

from dataclasses import dataclass
import wave
import io

try:
    from PIL import Image
except Exception as _e:
    Image = None
    _PIL_IMPORT_ERROR = _e
else:
    _PIL_IMPORT_ERROR = None

@dataclass
class CoverMetaAudio:
    """Metadata of an audio cover needed for LSB capacity math."""
    path: str
    channels: int                 # e.g., 1 (mono) or 2 (stereo)
    frames: int                   # total number of PCM frames
    bits_per_sample: int          # 8, 16, or 24 (PCM integer)
    sample_rate: int              # Hz (not used for capacity, but useful to display)
    is_pcm_integer: bool          # True if PCM integer (not float)

@dataclass
class CoverMetaImage:
    """Metadata of an image cover needed for LSB capacity math."""
    path: str
    width: int
    height: int
    channels: int
    mode: str
    bits_per_channel: int

def is_wav_extension(filename: str) -> bool:
    return filename.lower().endswith(".wav")

def is_image_extension(filename: str) -> bool:
    fn = filename.lower()
    return fn.endswith(".bmp") or fn.endswith(".png") or fn.endswith(".gif")

def load_audio_meta(path_or_bytes) -> CoverMetaAudio:
    """
    Reads a WAV header and returns audio metadata.
    Accepts a filesystem path (str) or bytes-like object.
    """
    def _read(wf):
        nch = wf.getnchannels()
        sampwidth_bytes = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        bits_per_sample = sampwidth_bytes * 8
        is_pcm_integer = bits_per_sample in (8, 16, 24)
        return nch, nframes, bits_per_sample, fr, is_pcm_integer

    if isinstance(path_or_bytes, (bytes, bytearray)):
        with wave.open(io.BytesIO(path_or_bytes), "rb") as wf:
            nch, nframes, bps, fr, is_pcm = _read(wf)
        return CoverMetaAudio(
            path="<bytes>",
            channels=nch,
            frames=nframes,
            bits_per_sample=bps,
            sample_rate=fr,
            is_pcm_integer=is_pcm,
        )
    else:
        with wave.open(path_or_bytes, "rb") as wf:
            nch, nframes, bps, fr, is_pcm = _read(wf)
        return CoverMetaAudio(
            path=path_or_bytes,
            channels=nch,
            frames=nframes,
            bits_per_sample=bps,
            sample_rate=fr,
            is_pcm_integer=is_pcm,
        )

def load_image_meta(path_or_bytes) -> CoverMetaImage:
    """
    Open an image (BMP/PNG/GIF) and return dimensions + channel count for capacity math.
    We treat palette/1-bit images as RGB for embedding capacity (3 channels).
    """
    if Image is None:
        raise RuntimeError(
            f"Pillow is required for image capacity checks but is not installed: {_PIL_IMPORT_ERROR}\n"
            f"Try: pip install pillow"
        )

    def _open_img(p):
        return Image.open(io.BytesIO(p)) if isinstance(p, (bytes, bytearray)) else Image.open(p)

    with _open_img(path_or_bytes) as img:
        mode = img.mode
        w, h = img.size

        # Decide channels used for stego capacity.
        # If grayscale ('L'): 1 channel; else treat as RGB (3 channels) for capacity purposes.
        if mode == "L":
            channels = 1
            bits_per_channel = 8
        else:
            # Convert-like capacity assumption for color images; your encoder will likely use RGB planes.
            channels = 3
            bits_per_channel = 8

        return CoverMetaImage(
            path="<bytes>" if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes),
            width=w,
            height=h,
            channels=channels,
            mode=mode,
            bits_per_channel=bits_per_channel,
        )

def compute_capacity_bytes(meta: CoverMetaAudio, lsb_count: int) -> int:
    """
    Capacity (bytes) available when using k LSBs on all samples of all channels.

    Math (audio):
        capacity_bytes = floor( frames * channels * lsb_count / 8 )
    """
    if lsb_count < 1 or lsb_count > 8:
        return 0
    return (meta.frames * meta.channels * lsb_count) // 8


def compute_capacity_bytes_image(meta: CoverMetaImage, lsb_count: int) -> int:
    """
    Capacity (bytes) for image covers using k LSBs on each channel of each pixel:
        capacity_bytes = floor( width * height * channels * lsb_count / 8 )
    """
    if lsb_count < 1 or lsb_count > 8:
        return 0
    return (meta.width * meta.height * meta.channels * lsb_count) // 8

def is_mp4_extension(filename: str) -> bool:
    return filename.lower().endswith(".mp4")

def load_mp4_meta(path_or_bytes) -> dict:
    """
    Minimal MP4 metadata for capacity. We treat capacity as file_size // 8.
    (i.e., we allow hiding ~12.5% of file size as payload in a custom box).
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        size = len(path_or_bytes)
    else:
        size = os.path.getsize(path_or_bytes)
    return {"path": "<bytes>" if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes),
            "size": size}

def compute_capacity_bytes_mp4(meta: dict, lsb_count: int = 1) -> int:
    # Reserve 1/8th of MP4 size to be safe
    return meta["size"] // 8
