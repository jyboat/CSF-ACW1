import uuid
import os
from flask import session
from PIL import Image
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from io import BytesIO

def detect_has_alpha(cover_bytes: bytes) -> bool:
    im = Image.open(BytesIO(cover_bytes))
    # Case 1: explicit alpha channel
    if im.mode in ("RGBA", "LA"):
        return True
    # Case 2 & 3: tRNS-based transparency (indexed or truecolor)
    if "transparency" in getattr(im, "info", {}):
        return True
    return False

def save_images_to_session(cover_bytes, stego_png, img_format):
    try:
        img_format = img_format.lstrip('.').lower()
        ext = '.' + img_format
        uid = uuid.uuid4().hex
        cover_filename = f"static/tmp/user/img/cover_{uid}{ext}"
        stego_filename = f"static/tmp/user/img/stego_{uid}{ext}"
        os.makedirs(os.path.dirname(cover_filename), exist_ok=True)  # <-- add
        with open(cover_filename, "wb") as f:
            f.write(cover_bytes)
        with open(stego_filename, "wb") as f:
            f.write(stego_png)
        session['cover'] = cover_filename.replace("static/", "")
        session['stego'] = stego_filename.replace("static/", "")
        return True
    except Exception as e:
        print(f"Error saving images to session: {e}")
        return False


def save_audio_to_session(cover_bytes, stego_wav):
    try:
        uid = uuid.uuid4().hex
        cover_filename = f"static/tmp/user/audio/cover_{uid}.wav"
        stego_filename = f"static/tmp/user/audio/stego_{uid}.wav"

        with open(cover_filename, "wb") as f:
            f.write(cover_bytes)

        with open(stego_filename, "wb") as f:
            f.write(stego_wav)
        
        session['cover'] = cover_filename.replace("static/", "")
        session['stego'] = stego_filename.replace("static/", "")
        return True
    
    except Exception as e:
        print(f"Error saving images to session: {e}")
        return False

def save_video_to_session(cover_bytes, stego_bytes):
    try:
        uid = uuid.uuid4().hex
        cover_filename = f"static/tmp/user/video/cover_{uid}.mp4"
        stego_filename = f"static/tmp/user/video/stego_{uid}.mp4"

        with open(cover_filename, "wb") as f:
            f.write(cover_bytes)
        with open(stego_filename, "wb") as f:
            f.write(stego_bytes)

        session['cover'] = cover_filename.replace("static/", "")
        session['stego'] = stego_filename.replace("static/", "")
        return True
    except Exception as e:
        print(f"Error saving videos to session: {e}")
        return False


def compute_pixel_diff(cover_path, stego_path):
    """
    Compare two image files pixel-by-pixel.
    Returns a list of coordinates where values differ.
    """
    cover_img = np.array(Image.open("static/" + cover_path).convert("RGB"))
    stego_img = np.array(Image.open("static/" + stego_path).convert("RGB"))

    if cover_img.shape != stego_img.shape:
        raise ValueError("Cover and stego images must have same dimensions")

    # Pixel-wise difference
    diff_mask = np.any(cover_img != stego_img, axis=-1)  # shape: (H, W), bool

    # get coordinates of changed pixels
    changed_coords = np.argwhere(diff_mask)  # shape: (N, 2), where each row is [y, x]

    # Convert to list of [x, y]
    diff_list = changed_coords[:, [1, 0]].tolist()
    return diff_list


def compute_audio_diff(cover_path, stego_path):
    """
    Compare two audio files sample-by-sample.
    Returns a list of sample indices (integers) where values differ.
    """

    cover_audio, sr1 = sf.read("static/" + cover_path)
    stego_audio, sr2 = sf.read("static/" + stego_path)

    if sr1 != sr2:
        raise ValueError("Sample rates differ between cover and stego audio")

    if cover_audio.shape != stego_audio.shape:
        raise ValueError("Cover and stego audio must have same shape")

    # Boolean mask of changed samples
    diff_mask = cover_audio != stego_audio

    # Get flat indices of changed samples
    changed_indices = np.argwhere(diff_mask).flatten()

    # Convert to Python list of ints
    diff_list = changed_indices.tolist()

    return diff_list


def save_rgb_analysis_to_session(cover_path, stego_path):
    """
    Fast RGB histogram overlay for large images.
    - Uses np.bincount (exact) instead of plt.hist.
    - Line/step plots (cheap to render).
    - Log y-scale so shapes remain visible for high pixel counts.
    """
    import os, uuid
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    cover_img = Image.open("static/" + cover_path).convert("RGB")
    stego_img = Image.open("static/" + stego_path).convert("RGB")

    cover = np.asarray(cover_img, dtype=np.uint8)
    stego = np.asarray(stego_img, dtype=np.uint8)

    if cover.shape != stego.shape:
        raise ValueError("Cover and stego images must have same dimensions")

    channel_names = ["Red", "Green", "Blue"]
    plt.figure(figsize=(12, 5), dpi=110)

    for i, name in enumerate(channel_names):
        ax = plt.subplot(1, 3, i+1)

        # Flatten channels
        c = cover[..., i].ravel()
        s = stego[..., i].ravel()

        # Exact 256-bin hist via bincount (much faster than plt.hist).
        c_counts = np.bincount(c, minlength=256)
        s_counts = np.bincount(s, minlength=256)

        x = np.arange(256)

        # Draw stego on top so it's visible; outline-ish using step.
        ax.step(x, s_counts, where="mid", label="Stego")
        ax.step(x, c_counts, where="mid", label="Cover")

        ax.set_title(f"{name} channel")
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("No. of pixels")
        ax.set_xlim(0, 255)

        # Log scale helps when counts are huge.
        ax.set_yscale("log")
        ax.legend()

    plt.tight_layout()
    uid = uuid.uuid4().hex
    out_path = f"static/tmp/user/img/rgb_analysis_{uid}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close("all")

    # Free memory explicitly for giant images
    del cover, stego, cover_img, stego_img

    session['rgb_analysis_filepath'] = out_path.replace("static/", "")
    return True


def save_audio_analysis_to_session(cover_path, stego_path):
    """
    Perform basic steganalysis for audio using matplotlib.
    Produces difference signal.
    """

    # Load both audio files
    cover_audio, sr1 = sf.read("static/" + cover_path)
    stego_audio, sr2 = sf.read("static/" + stego_path)

    if sr1 != sr2:
        raise ValueError("Sample rates differ between cover and stego audio")
    if cover_audio.shape != stego_audio.shape:
        raise ValueError("Cover and stego audio must have same shape")

    # Difference signal
    diff_signal = stego_audio - cover_audio

    # Limit plotting window for clarity (first N samples)
    n = len(cover_audio)
    x = np.arange(n) / sr1  # time axis in seconds

    plt.figure(figsize=(12, 4))

    # 1. Difference signal
    ax1 = plt.subplot(3, 1, 2)
    ax1.plot(x, diff_signal[:n], color="red", linewidth=0.7)
    ax1.set_title("Difference Signal (Stego - Cover)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")

    plt.tight_layout()

    # Save plot to session
    uid = uuid.uuid4().hex
    out_path = f"static/tmp/user/audio/audio_analysis_{uid}.png"
    plt.savefig(out_path)
    plt.close()

    session["audio_analysis_filepath"] = out_path.replace("static/", "")

    return True

def save_image_comparison_to_session(cover_path: str, stego_path: str) -> bool:
    """
    Side-by-side comparison with a background-aware difference heatmap:
    - Load as RGBA (preserve transparency)
    - Compute RGB difference
    - Mask pixels that are fully transparent in BOTH images
    - Render masked pixels transparent in the heatmap
    """

    cover_rgba = np.array(Image.open("static/" + cover_path).convert("RGBA"))
    stego_rgba = np.array(Image.open("static/" + stego_path).convert("RGBA"))

    if cover_rgba.shape != stego_rgba.shape:
        raise ValueError("Cover and stego images must have same dimensions")

    # Split channels
    c_rgb, c_a = cover_rgba[..., :3].astype(np.int16), cover_rgba[..., 3].astype(np.uint8)
    s_rgb, s_a = stego_rgba[..., :3].astype(np.int16), stego_rgba[..., 3].astype(np.uint8)

    # Difference on RGB channels
    diff = np.abs(s_rgb - c_rgb).mean(axis=2).astype(np.float32)  # 0..255

    # Mask: ignore background where BOTH are fully transparent
    bg_mask = (c_a == 0) & (s_a == 0)
    diff_masked = np.ma.array(diff, mask=bg_mask)

    # Contrast stretch on non-masked values
    if diff_masked.count() > 0:
        vals = diff_masked.compressed()
        p_low, p_high = np.percentile(vals, [1, 99])
        if p_high <= p_low:
            p_low, p_high = 0.0, 1.0
        diff_vis = np.ma.clip((diff_masked - p_low) / (p_high - p_low), 0, 1)
    else:
        diff_vis = diff_masked  # all masked

    # Colormap with transparent for masked (background) pixels
    cmap = plt.cm.magma.copy()
    cmap.set_bad(alpha=0.0)

    # For the side-by-side cover/stego previews, composite RGBA on white so they look natural
    def composite_on_white(rgba):
        rgb = rgba[..., :3].astype(np.float32)
        a = (rgba[..., 3:4].astype(np.float32)) / 255.0
        return (rgb * a + 255.0 * (1.0 - a)).astype(np.uint8)

    cover_vis = composite_on_white(cover_rgba)
    stego_vis = composite_on_white(stego_rgba)

    # ---- Plot ----
    plt.figure(figsize=(12, 4), dpi=110)

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(cover_vis)
    ax1.set_title("Cover")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(stego_vis)
    ax2.set_title("Stego")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(diff_vis, cmap=cmap, vmin=0, vmax=1)
    ax3.set_title("Difference highlighted in pink")
    ax3.axis("off")

    plt.tight_layout()
    uid = uuid.uuid4().hex
    out_path = f"static/tmp/user/img/diff_visual_{uid}.png"
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close()

    session["diff_visualization_filepath"] = out_path.replace("static/", "")
    return True

def save_gray_analysis_to_session(cover_path, stego_path):
    """
    Fast grayscale histogram overlay for large images.
    - np.bincount for exact binning
    - step lines (cheap)
    - log y-scale for visibility
    """
    import os, uuid
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    cover_gray = np.asarray(Image.open("static/" + cover_path).convert("L"), dtype=np.uint8)
    stego_gray = np.asarray(Image.open("static/" + stego_path).convert("L"), dtype=np.uint8)

    if cover_gray.shape != stego_gray.shape:
        raise ValueError("Cover and stego images must have same dimensions")

    c_counts = np.bincount(cover_gray.ravel(), minlength=256)
    s_counts = np.bincount(stego_gray.ravel(), minlength=256)
    x = np.arange(256)

    plt.figure(figsize=(8, 4.5), dpi=110)
    plt.step(x, s_counts, where="mid", label="Stego")
    plt.step(x, c_counts, where="mid", label="Cover")

    plt.title("Grayscale Intensity Distribution")
    plt.xlabel("Pixel intensity (0–255)")
    plt.ylabel("No. of pixels")
    plt.xlim(0, 255)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    uid = uuid.uuid4().hex
    out_path = f"static/tmp/user/img/gray_analysis_{uid}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close("all")

    # Free memory
    del cover_gray, stego_gray

    session['gray_analysis_filepath'] = out_path.replace("static/", "")
    return True

def save_spectrogram_comparison_to_session(cover_path, stego_path):
    # Load both audio files
    cover_audio, sr1 = sf.read("static/" + cover_path)
    stego_audio, sr2 = sf.read("static/" + stego_path)

    if sr1 != sr2:
        raise ValueError("Sample rates differ between cover and stego audio")
    if cover_audio.shape != stego_audio.shape:
        raise ValueError("Cover and stego audio must have same shape")
    
    # If 
    if cover_audio.ndim > 1:
        cover_audio = cover_audio[:,0]
    if stego_audio.ndim > 1:
        stego_audio = stego_audio[:,0]

    # Difference signal
    diff_signal = stego_audio - cover_audio

    # Plot spectrograms side by side
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.specgram(cover_audio, Fs=sr1, cmap="inferno")
    plt.title("Cover Audio Spectogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.subplot(1, 3, 2)
    plt.specgram(stego_audio, Fs=sr1, cmap="inferno")
    plt.title("Stego Audio Spectogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.subplot(1, 3, 3)
    plt.specgram(diff_signal, Fs=sr1, cmap="inferno")
    plt.title("Stego-Cover Spectrogram Difference")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")


    plt.tight_layout()

    # Save spectogram image
    uid = uuid.uuid4().hex
    out_path = f"static/tmp/user/audio/spectrogram_{uid}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    # Save path to session
    session["audio_spectrogram_filepath"] = out_path.replace("static/", "")
    return True


def save_gif_diff_animation_to_session(cover_path: str, stego_path: str) -> bool:
    """
    Build an animated GIF showing per-frame differences (heatmap-style).
    Saves to static/tmp/user/img/gif_diff_<uuid>.gif and sets session['gif_diff_filepath'].
    """
    import os, uuid
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    cover_file = "static/" + cover_path
    stego_file = "static/" + stego_path

    with Image.open(cover_file) as c, Image.open(stego_file) as s:
        if not (getattr(c, "is_animated", False) and getattr(s, "is_animated", False)):
            raise ValueError("Both cover and stego must be animated GIFs")
        if c.n_frames != s.n_frames:
            raise ValueError("GIFs must have the same number of frames")

        frames_out = []
        durations = []
        loop = c.info.get("loop", 0)

        for i in range(c.n_frames):
            c.seek(i); s.seek(i)
            c_rgba = c.convert("RGBA")
            s_rgba = s.convert("RGBA")

            if c_rgba.size != s_rgba.size:
                raise ValueError(f"Frame {i} sizes differ")

            c_arr = np.array(c_rgba, dtype=np.int16)[..., :3]
            s_arr = np.array(s_rgba, dtype=np.int16)[..., :3]

            # Mean absolute diff per pixel -> 0..255
            diff = np.abs(s_arr - c_arr).mean(axis=2).astype(np.float32)

            # Contrast stretch so small diffs are visible
            if diff.size:
                p1, p99 = np.percentile(diff, [1, 99])
                if p99 <= p1:
                    p1, p99 = 0.0, 1.0
                vis = np.clip((diff - p1) / (p99 - p1), 0, 1)
            else:
                vis = diff

            rgb = (plt.cm.magma(vis)[..., :3] * 255).astype(np.uint8)
            frames_out.append(Image.fromarray(rgb, mode="RGB"))
            durations.append(c.info.get("duration", 100))

    uid = uuid.uuid4().hex
    out_path = f"static/tmp/user/img/gif_diff_{uid}.gif"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    frames_out[0].save(
        out_path, save_all=True, append_images=frames_out[1:],
        duration=durations, loop=loop, optimize=False, disposal=2
    )

    session["gif_diff_filepath"] = out_path.replace("static/", "")
    return True
