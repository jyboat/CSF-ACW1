import base64
import uuid
from flask import session
from PIL import Image
import numpy as np

def save_images_to_session(cover_bytes, stego_png):
    try:
        uid = uuid.uuid4().hex
        cover_filename = f"static/tmp/user/img/cover_{uid}.png"
        stego_filename = f"static/tmp/user/img/stego_{uid}.png"

        with open(cover_filename, "wb") as f:
            f.write(cover_bytes)

        with open(stego_filename, "wb") as f:
            f.write(stego_png)
        
        session['cover_image'] = cover_filename.replace("static/", "")
        session['stego_image'] = stego_filename.replace("static/", "")
        return True
    
    except Exception as e:
        print(f"Error saving images to session: {e}")
        return False
    
def compute_pixel_diff(cover_path, stego_path):
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