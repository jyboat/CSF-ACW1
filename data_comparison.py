import uuid
from flask import session
from PIL import Image
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def save_images_to_session(cover_bytes, stego_png):
    try:
        uid = uuid.uuid4().hex
        cover_filename = f"static/tmp/user/img/cover_{uid}.png"
        stego_filename = f"static/tmp/user/img/stego_{uid}.png"

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
    cover_img = np.array(Image.open("static/" + cover_path).convert("RGB"))
    stego_img = np.array(Image.open("static/" + stego_path).convert("RGB"))

    channel_names = ["Red", "Green", "Blue"]
    colors = ["red", "green", "blue"]

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
                histtype="stepfilled", linewidth=1.0, label="Cover", color="black",)

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


def save_audio_analysis_to_session(cover_path, stego_path, window=5000):
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
    n = min(window, len(cover_audio))
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
