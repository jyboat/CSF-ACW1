import uuid
import os
import cv2
from flask import session
from PIL import Image
from skimage.metrics import structural_similarity as ssim
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
        img_format = img_format.lstrip('.').lower()  # e.g. 'png', 'jpeg'
        ext = '.' + img_format
    
        uid = uuid.uuid4().hex
        cover_filename = f"static/tmp/user/img/cover_{uid}{ext}"
        stego_filename = f"static/tmp/user/img/stego_{uid}{ext}"

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


def compute_video_diff(cover_path, stego_path, step=60, ssim_threshold=0.99):
    """
    Compare two video files using SSIM on sampled frames.
    Returns a list of (frame_idx, ssim_score) for frames below the threshold.
    """

    cap_cover = cv2.VideoCapture("static/" + cover_path)
    cap_stego = cv2.VideoCapture("static/" + stego_path)

    if not cap_cover.isOpened() or not cap_stego.isOpened():
        raise ValueError("Could not open one or both video files")

    diff_frames = []
    idx = 0

    while True:
        ret1, frame1 = cap_cover.read()
        ret2, frame2 = cap_stego.read()
        if not ret1 or not ret2:
            break

        if frame1.shape != frame2.shape:
            raise ValueError("Cover and stego frames must have same dimensions")

        # Only check every Nth frame
        if idx % step == 0:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            score, _ = ssim(gray1, gray2, full=True)
            if score < ssim_threshold:
                diff_frames.append((idx, score))

        idx += 1

    cap_cover.release()
    cap_stego.release()

    return diff_frames


def save_rgb_analysis_to_session(cover_path, stego_path):
    cover_img = np.array(Image.open("static/" + cover_path).convert("RGB"))
    stego_img = np.array(Image.open("static/" + stego_path).convert("RGB"))

    channel_names = ["Red", "Green", "Blue"]
    # colors = ["red", "green", "blue"]

    plt.figure(figsize=(12, 5))
    
    # indexes the channel (0=Red, 1=Green, 2=Blue)
    for i, name in enumerate(channel_names):
        ax = plt.subplot(1, 3, i+1)
        c = cover_img[..., i].ravel()
        s = stego_img[..., i].ravel()

        # Stego: 
        ax.hist(s, bins=np.arange(257), range=(0, 256),
                histtype="stepfilled", alpha=1.0, label="Stego", color="yellow")

        # Cover: 
        ax.hist(c, bins=np.arange(257), range=(0, 256),
                histtype="stepfilled", linewidth=1.0, label="Cover", color="purple",)

        ax.set_title(f"{name} channel")
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("No. of pixels")
        ax.set_xlim(0, 255)
        ax.legend()

    plt.tight_layout()
    uid = uuid.uuid4().hex
    out_path = f"static/tmp/user/img/rgb_analysis_{uid}.png"
    plt.savefig(out_path)
    plt.close()

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
    Build a grayscale/luminance intensity histogram:
    - Convert both images to 'L' (8-bit grayscale)
    - Overlay histograms for Cover vs Stego
    - Saves to static/tmp/user/img/gray_analysis_<uuid>.png
    - Stores path in session['gray_analysis_filepath']
    """

    cover_gray = np.array(Image.open("static/" + cover_path).convert("L"))
    stego_gray = np.array(Image.open("static/" + stego_path).convert("L"))

    if cover_gray.shape != stego_gray.shape:
        raise ValueError("Cover and stego images must have same dimensions")

    c = cover_gray.ravel()
    s = stego_gray.ravel()

    plt.figure(figsize=(8, 4.5), dpi=110)

    # Stego on top (filled), Cover beneath (outlined) – consistent with your RGB plot
    plt.hist(s, bins=np.arange(257), range=(0, 256),
             histtype="stepfilled", alpha=1.0, label="Stego", color="yellow")
    plt.hist(c, bins=np.arange(257), range=(0, 256),
             histtype="stepfilled", linewidth=1.2, label="Cover", color="purple")

    plt.title("Grayscale Intensity Distribution")
    plt.xlabel("Pixel intensity (0–255)")
    plt.ylabel("No. of pixels")
    plt.xlim(0, 255)
    plt.legend()
    plt.tight_layout()

    uid = uuid.uuid4().hex
    out_path = f"static/tmp/user/img/gray_analysis_{uid}.png"
    plt.savefig(out_path)
    plt.close()

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


def save_video_psnr_analysis_to_session(cover_path, stego_path, step=60):
    """
    Perform analysis of cover vs stego video:
    - Compute PSNR values on sampled frames (every 'step')
    - Save a PSNR-over-frames plot to session
    """

    cap_cover = cv2.VideoCapture("static/" + cover_path)
    cap_stego = cv2.VideoCapture("static/" + stego_path)

    if not cap_cover.isOpened() or not cap_stego.isOpened():
        raise ValueError("Could not open one or both video files")

    def mse(img1, img2):
        return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

    def psnr(img1, img2):
        mse_val = mse(img1, img2)
        if mse_val == 0:
            return float("inf")
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse_val))

    psnr_values = []
    frame_indices = []
    idx = 0

    while True:
        ret1, frame1 = cap_cover.read()
        ret2, frame2 = cap_stego.read()
        if not ret1 or not ret2:
            break

        if frame1.shape != frame2.shape:
            raise ValueError("Cover and stego frames must have same dimensions")

        # Only compute every Nth frame
        if idx % step == 0:
            psnr_values.append(psnr(frame1, frame2))
            frame_indices.append(idx)

        idx += 1

    cap_cover.release()
    cap_stego.release()

    # ---- Plot PSNR over frames ----
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, psnr_values, color="blue", marker="o", linestyle="-")
    plt.xlabel("Frame Index")
    plt.ylabel("PSNR (dB)")
    plt.title(f"PSNR Across Frames (Sampled every {step} frames)")
    plt.grid(True)

    uid = uuid.uuid4().hex
    out_path = f"static/tmp/user/video/video_psnr_{uid}.png"
    plt.savefig(out_path)
    plt.close()

    session['video_psnr_filepath'] = out_path.replace("static/", "")
    return True

