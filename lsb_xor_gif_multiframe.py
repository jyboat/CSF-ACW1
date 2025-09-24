import io
from dataclasses import dataclass
import numpy as np
from PIL import Image

from data_comparison import (
    detect_has_alpha,
    )

# lsb_xor_algorithm.py  
from lsb_xor_algorithm import (
    embed_xor_lsb_from_xy,              
    extract_xor_lsb_auto,
    _sobel_magnitude      
)

# Multi-frame LSB+XOR for animated GIFs
def gif_to_combined_flat(gif_bytes: bytes, mode: str = "RGB"):
    """Convert all GIF frames into one giant flat array"""
    im = Image.open(io.BytesIO(gif_bytes))
    
    if not (hasattr(im, 'is_animated') and im.is_animated and im.n_frames > 1):
        raise ValueError("Not an animated GIF")
    
    frames = []
    durations = []
    
    for i in range(im.n_frames):
        im.seek(i)
        frame = im.copy().convert(mode)
        frames.append(np.array(frame, dtype=np.uint8))
        durations.append(im.info.get('duration', 100))
    
    # Concatenate all frames into one massive flat array
    combined_flat = np.concatenate([frame.reshape(-1) for frame in frames])
    
    # Metadata needed for reconstruction
    frame_shape = frames[0].shape
    n_frames = len(frames)
    pixels_per_frame = frames[0].size
    
    meta = {
        'durations': durations,
        'loop': im.info.get('loop', 0),
        'frame_shape': frame_shape,
        'n_frames': n_frames,
        'pixels_per_frame': pixels_per_frame,
        'mode': mode
    }
    
    return combined_flat, meta

def combined_flat_to_gif(combined_flat: np.ndarray, meta: dict) -> bytes:
    """Reconstruct animated GIF from combined flat array"""
    frame_shape = meta['frame_shape']
    n_frames = meta['n_frames']
    pixels_per_frame = meta['pixels_per_frame']
    mode = meta['mode']
    
    # Split combined array back into individual frames
    frames = []
    for i in range(n_frames):
        start_idx = i * pixels_per_frame
        end_idx = start_idx + pixels_per_frame
        frame_flat = combined_flat[start_idx:end_idx]
        frame = frame_flat.reshape(frame_shape).astype(np.uint8)
        frames.append(Image.fromarray(frame, mode=mode))
    
    # Save as animated GIF
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=meta['durations'],
        loop=meta.get('loop', 0),
        optimize=False
    )
    buf.seek(0)
    return buf.getvalue()

def embed_gif_multiframe_lsb_xor(gif_bytes: bytes, payload: bytes, k: int, key: str) -> bytes:
    """Embed payload across ALL frames of animated GIF using LSB+XOR"""
    
    # Detect mode
    has_alpha = detect_has_alpha(gif_bytes)
    mode = "RGBA" if has_alpha else "RGB"
    
    # Convert all frames to one giant flat array
    combined_flat, meta = gif_to_combined_flat(gif_bytes, mode)
    
    # Create a "virtual shape" for the combined array
    # Treat it as a single massive "image"
    virtual_height = meta['n_frames']
    virtual_width = meta['pixels_per_frame']
    virtual_shape = (virtual_height, virtual_width) if len(meta['frame_shape']) == 2 else (virtual_height, virtual_width, meta['frame_shape'][2])
    
    # Existing LSB+XOR embedding for the whole img. Use simple start (0,0)
    stego_flat = embed_xor_lsb_from_xy(
        combined_flat, virtual_shape, payload, k, key, 
        start_x=0, start_y=0
    )
    
    # Reconstruct animated GIF from modified flat array
    return combined_flat_to_gif(stego_flat, meta)

def extract_gif_multiframe_lsb_xor(gif_bytes: bytes, k: int, key: str) -> bytes:
    """Extract payload from ALL frames of animated GIF"""
    
    # Detect mode
    has_alpha = detect_has_alpha(gif_bytes) 
    mode = "RGBA" if has_alpha else "RGB"
    
    # Convert all frames to combined flat array
    combined_flat, meta = gif_to_combined_flat(gif_bytes, mode)
    
    # Create virtual shape
    virtual_height = meta['n_frames']
    virtual_width = meta['pixels_per_frame']
    virtual_shape = (virtual_height, virtual_width) if len(meta['frame_shape']) == 2 else (virtual_height, virtual_width, meta['frame_shape'][2])

    # Use existing LSB+XOR extraction
    return extract_xor_lsb_auto(combined_flat, virtual_shape, k, key)

def compute_gif_multiframe_capacity(gif_bytes: bytes, k: int) -> int:
    """Compute LSB capacity across ALL frames"""
    im = Image.open(io.BytesIO(gif_bytes))
    
    if not (hasattr(im, 'is_animated') and im.is_animated):
        raise ValueError("Not an animated GIF")
    
    # Get first frame info
    first_frame = im.copy().convert("RGB")  
    width, height = first_frame.size
    channels = len(first_frame.getbands())
    n_frames = im.n_frames
    
    # Total pixels across ALL frames
    total_pixels = width * height * channels * n_frames
    
    # Capacity calculation (minus header overhead)
    header_bits = 24 * 8  # STG2 header size
    available_bits = (total_pixels * k) - header_bits
    
    return max(0, available_bits // 8)
