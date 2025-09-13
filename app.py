import os
from flask import Flask, render_template, url_for, redirect, request, flash
from flask_toastr import Toastr
from werkzeug.utils import secure_filename
from capacity import is_wav_extension, load_audio_meta, compute_capacity_bytes, is_image_extension, load_image_meta, compute_capacity_bytes_image
from typing import Any
import io

app = Flask(__name__)
toastr = Toastr(app)

# Flask session secret; required for flash messaging
app.config["SECRET_KEY"] = "dev-key"

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


@app.route('/', methods=['GET', 'POST'])
def index():
    """
       GET: Renders form with two steps (your template).
       POST: Accepts audio cover (WAV) and payload (any file), plus (lsb, key).
             Validates inputs and computes capacity for the WAV cover only.

       POST form fields (types):
           cover: FileStorage (required) -> must be .wav and valid PCM integer WAV (8/16/24-bit)
           payload: FileStorage (required) -> any file type, just measured by size
           lsb: str (required) -> integer 1..8
           key: str (required) -> integer (not used here yet, but spec requires it later)

       Response context (types):
           capacity: int | None
           fits: bool | None
           cover_meta: CoverMetaAudio | None
           payload_size: int | None
           lsb: int | None
           key: int | None
           cover_filename: str | None
           payload_filename: str | None
       """

    context: dict[str, Any] = {
        "capacity": None,
        "fits": None,
        "cover_meta": None,
        "payload_size": None,
        "lsbCount": None,
        "stegoKey": None,
        "cover_filename": None,
        "payload_filename": None,
    }

    if request.method == 'POST':
        cover = request.files.get("coverFile")
        payload = request.files.get("payload")
        lsb_str = request.form.get("lsbCount", "").strip()
        key_str = request.form.get("stegoKey", "").strip()

        # --- Basic field presence checks --------------------------------------
        if not cover or not cover.filename:
            flash("Please upload a cover file (.wav, .bmp, .png, or .gif).", "error")
            return render_template('index.html', **context)
        if not payload or not payload.filename:
            flash("Please upload a payload file.", "error")
            return render_template('index.html', **context)

        # --- LSB count validation (must be 1..8 per spec) ----------------------
        try:
            lsb = int(lsb_str)
        except ValueError:
            flash("LSB count must be an integer (1–8).", "error")
            return render_template('index.html', **context)
        if lsb < 1 or lsb > 8:
            flash("LSB count must be between 1 and 8.", "error")
            return render_template('index.html', **context)

        # --- Key validation (numeric required by spec, used later for encode) ---
        try:
            key = int(key_str)
        except ValueError:
            flash("Key must be an integer (required).", "error")
            return render_template('index.html', **context)

        # --- Persist files to disk ---------------------------------------------
        cover_bytes = cover.read()  # bytes
        payload_bytes = payload.read()  # bytes
        payload_size = len(payload_bytes)  # int
        context["cover_filename"] = cover.filename
        context["payload_filename"] = payload.filename


        # --- WAV header validation & meta (PCM integer check) -------------------
        try:
            if is_wav_extension(cover.filename):
                # --- AUDIO path ---
                meta = load_audio_meta(cover_bytes)  # CoverMetaAudio
                if not meta.is_pcm_integer:
                    flash("Unsupported WAV format. Please provide PCM 8/16/24-bit WAV (not float).", "error")
                    return render_template('index.html', **context)

                capacity_bytes = compute_capacity_bytes(meta, lsb)
                meta_obj = meta  # store if you want to display later

            elif is_image_extension(cover.filename):
                # --- IMAGE path ---
                meta_img = load_image_meta(cover_bytes)  # CoverMetaImage
                capacity_bytes = compute_capacity_bytes_image(meta_img, lsb)
                meta_obj = meta_img

            else:
                flash("Unsupported cover type. Use a .wav (audio) or .bmp/.png/.gif (image).", "error")
                return render_template('index.html', **context)

        except Exception as e:
            flash(f"Failed to read cover header: {e}", "error")
            return render_template('index.html', **context)

        fits = payload_size <= capacity_bytes

        context.update({
            "capacity": capacity_bytes,
            "fits": fits,
            "cover_meta": meta_obj,
            "payload_size": payload_size,
            "lsbCount": lsb,
            "stegoKey": key,
        })

        if not fits:
            flash(
                f"Payload too large for this cover at {lsb} LSB(s). "
                f"Payload={payload_size} bytes, Cover File={capacity_bytes} bytes.",
                "error"
            )
        else:
            flash(
                f"Capacity OK. Cover File={capacity_bytes} bytes, Payload={payload_size} bytes.",
                "success"
            )

        return render_template('index.html', **context)

    return render_template('index.html', **context)

@app.route('/results')
def results():
    return render_template('results.html')


if __name__ == "__main__":
    app.debug = True
    app.run()
