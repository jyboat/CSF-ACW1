from flask_toastr import Toastr
from capacity import is_wav_extension, load_audio_meta, compute_capacity_bytes, is_image_extension, load_image_meta, compute_capacity_bytes_image
from typing import Any
import io
import os
from flask import (
    Flask, request, render_template, redirect, url_for,
    flash, send_file, jsonify
)
from capacity import (
    is_wav_extension,
    is_image_extension,
    load_audio_meta,
    load_image_meta,
    compute_capacity_bytes,
    compute_capacity_bytes_image,
)
from lsb_xor_algorithm import (
    select_complex_indices_from_image,
    embed_xor_lsb_at_indices,
    extract_xor_lsb_at_indices,
    flat_to_image,
    image_to_flat
)

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

@app.route("/check", methods=["POST"])
def check_capacity_form():
    """
    Capacity check from the main form (returns to the page with a flash).
    Works for both images and WAV.
    """
    cover = request.files.get("coverFile")
    lsb_str = request.form.get("lsbCount", "1")

    if not cover or not cover.filename:
        flash("Please upload a cover file.", "error")
        return redirect(url_for("index"))

    try:
        lsb = int(lsb_str)
        if not (1 <= lsb <= 8):
            raise ValueError
    except Exception:
        flash("Invalid LSB count. Use a number between 1 and 8.", "error")
        return redirect(url_for("index"))

    cover_bytes = cover.read()
    try:
        if is_wav_extension(cover.filename):
            meta = load_audio_meta(cover_bytes)
            if not meta.is_pcm_integer:
                flash("WAV must be 8/16/24-bit PCM.", "error")
                return redirect(url_for("index"))
            cap = compute_capacity_bytes(meta, lsb)
            flash(f"Audio capacity: {cap} bytes available.", "success")
        elif is_image_extension(cover.filename):
            meta_img = load_image_meta(cover_bytes)
            cap = compute_capacity_bytes_image(meta_img, lsb)
            flash(f"Image capacity: {cap} bytes available (theoretical).", "success")
        else:
            flash("Unsupported cover type. Use PNG/BMP/GIF for images or WAV for audio.", "error")
    except Exception as e:
        flash(f"Capacity check failed: {e}", "error")

    return redirect(url_for("index"))


@app.route("/api/check-capacity", methods=["POST"])
def api_check_capacity():
    """
    JSON capacity endpoint (useful for AJAX). Image & WAV.
    """
    cover = request.files.get("coverFile")
    lsb_str = request.form.get("lsbCount", "1")
    try:
        lsb = int(lsb_str)
        if not (1 <= lsb <= 8):
            raise ValueError
    except Exception:
        return jsonify({"ok": False, "error": "Invalid LSB (1..8)"}), 400

    if not cover or not cover.filename:
        return jsonify({"ok": False, "error": "Missing cover"}), 400

    cover_bytes = cover.read()
    try:
        if is_wav_extension(cover.filename):
            meta = load_audio_meta(cover_bytes)
            if not meta.is_pcm_integer:
                return jsonify({"ok": False, "error": "WAV must be 8/16/24-bit PCM"}), 400
            cap = compute_capacity_bytes(meta, lsb)
        elif is_image_extension(cover.filename):
            meta_img = load_image_meta(cover_bytes)
            cap = compute_capacity_bytes_image(meta_img, lsb)
        else:
            return jsonify({"ok": False, "error": "Unsupported cover type"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "capacity_bytes": cap})


@app.route("/embed", methods=["POST"])
def embed_image():
    """
    Embed payload into an image using your LSB+XOR engine with complex-pixel selection.
    Produces a PNG stego (lossless).
    """
    cover = request.files.get("coverFile")
    lsb_str = request.form.get("lsbCount", "1")
    key = request.form.get("stegoKey", "").strip()
    text_payload_str = request.form.get("textPayload", "")
    payload_file = next((f for f in request.files.getlist("payloadFile") if f and f.filename), None)

    if not cover or not cover.filename:
        flash("Upload a cover image first.", "error")
        return redirect(url_for("index"))

    try:
        lsb = int(lsb_str)
        if not (1 <= lsb <= 8):
            raise ValueError
    except Exception:
        flash("Invalid LSB count. Use a number between 1 and 8.", "error")
        return redirect(url_for("index"))

    if not key:
        flash("Key is required.", "error")
        return redirect(url_for("index"))

    # Accept either text or file payload
    if text_payload_str and payload_file:
        flash("Choose text OR a file (not both).", "error")
        return redirect(url_for("index"))
    if not text_payload_str and not payload_file:
        flash("Provide a payload (text or file).", "error")
        return redirect(url_for("index"))

    payload_bytes = text_payload_str.encode("utf-8") if text_payload_str else payload_file.read()

    # Images only for embed (audio later)
    if not is_image_extension(cover.filename):
        flash("Embed currently supports images (.png/.bmp/.gif).", "error")
        return redirect(url_for("index"))

    cover_bytes = cover.read()
    try:
        # Capacity (theoretical)
        meta_img = load_image_meta(cover_bytes)
        capacity_bytes = compute_capacity_bytes_image(meta_img, lsb)
        if len(payload_bytes) > capacity_bytes:
            flash(f"Payload too large: {len(payload_bytes)} > {capacity_bytes} bytes (theoretical).", "error")
            return redirect(url_for("index"))

        # Prepare pixels and select complex indices (same recipe must be used at extract)
        flat_cover, shape, _ = image_to_flat(cover_bytes, mode="RGB")
        # Using top 30% complex pixels; keep this consistent at extract
        _, _, _, eligible = select_complex_indices_from_image(
            cover_bytes, top_percent=30, mode="RGB", key=key
        )

        # We need bits for header (16 bytes) + payload
        bits_needed = (16 + len(payload_bytes)) * 8
        bits_avail = len(eligible) * lsb
        if bits_needed > bits_avail:
            flash(
                "Not enough complex locations with current LSB/top% settings. "
                f"Need {bits_needed} bits, have {bits_avail} bits.",
                "error"
            )
            return redirect(url_for("index"))

        # Embed
        stego_flat = embed_xor_lsb_at_indices(
            flat_cover, payload_bytes, k=lsb, key=key, indices=eligible
        )
        stego_png = flat_to_image(stego_flat, shape, mode="RGB")

        return send_file(
            io.BytesIO(stego_png),
            mimetype="image/png",
            as_attachment=True,
            download_name="stego.png",
        )
    except Exception as e:
        flash(f"Embed failed: {e}", "error")
        return redirect(url_for("index"))


@app.route("/detect", methods=["POST"])
def detect_header():
    """
    Try to detect a valid STG1 header/checksum in a suspected stego image.
    """
    stego = request.files.get("stegoFile")
    lsb_str = request.form.get("lsbCount", "1")
    key = request.form.get("stegoKey", "").strip()

    if not stego or not stego.filename:
        flash("Upload a suspected stego image.", "error")
        return redirect(url_for("index"))

    try:
        lsb = int(lsb_str)
        if not (1 <= lsb <= 8):
            raise ValueError
    except Exception:
        flash("Invalid LSB count. Use a number between 1 and 8.", "error")
        return redirect(url_for("index"))

    if not key:
        flash("Key required for detection.", "error")
        return redirect(url_for("index"))

    if not is_image_extension(stego.filename):
        flash("Detection currently supports images (.png/.bmp/.gif).", "error")
        return redirect(url_for("index"))

    img_bytes = stego.read()
    try:
        flat, _, _ = image_to_flat(img_bytes, mode="RGB")
        _, _, _, eligible = select_complex_indices_from_image(
            img_bytes, top_percent=30, mode="RGB", key=key
        )

        # Header is 16 bytes => 128 bits. Need ceil(128 / lsb) pixels from eligible.
        header_bits = 16 * 8
        groups_hdr = (header_bits + lsb - 1) // lsb
        if groups_hdr > len(eligible):
            flash("Not enough complex locations to read header.", "error")
            return redirect(url_for("index"))

        # Full extract attempt; the extractor validates header & checksum internally.
        _ = extract_xor_lsb_at_indices(flat, k=lsb, key=key, indices=eligible)
        flash("Likely stego found (valid STG1 header and checksum).", "success")
    except Exception as e:
        flash(f"No valid header detected or wrong key/LSB: {e}", "error")

    return redirect(url_for("index"))


@app.route("/extract", methods=["POST"])
def extract_image():
    """
    Extract an embedded payload from an image using the same parameters (LSB/key).
    """
    stego = request.files.get("stegoFile")
    lsb_str = request.form.get("lsbCount", "1")
    key = request.form.get("stegoKey", "").strip()

    if not stego or not stego.filename:
        flash("Upload a stego image.", "error")
        return redirect(url_for("index"))

    try:
        lsb = int(lsb_str)
        if not (1 <= lsb <= 8):
            raise ValueError
    except Exception:
        flash("Invalid LSB count. Use a number between 1 and 8.", "error")
        return redirect(url_for("index"))

    if not key:
        flash("Key is required.", "error")
        return redirect(url_for("index"))

    if not is_image_extension(stego.filename):
        flash("Extraction currently supports images (.png/.bmp/.gif).", "error")
        return redirect(url_for("index"))

    img_bytes = stego.read()
    try:
        flat, _, _ = image_to_flat(img_bytes, mode="RGB")
        _, _, _, eligible = select_complex_indices_from_image(
            img_bytes, top_percent=30, mode="RGB", key=key
        )
        payload = extract_xor_lsb_at_indices(flat, k=lsb, key=key, indices=eligible)

        return send_file(
            io.BytesIO(payload),
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="extracted_payload.bin",
        )
    except Exception as e:
        flash(f"Extract failed: {e}", "error")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.debug = True
    app.run()
