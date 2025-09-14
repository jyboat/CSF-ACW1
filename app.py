from flask_toastr import Toastr
from flask import (
    Flask, request, render_template, redirect, url_for,
    flash, send_file, jsonify, session
)
from typing import Any
import io
import os
import wave
import numpy as np

# capacity.py
from capacity import (
    is_wav_extension,
    is_image_extension,
    load_audio_meta,
    load_image_meta,
    compute_capacity_bytes,
    compute_capacity_bytes_image,
)

# data_comparison helpers
from data_comparison import save_images_to_session, compute_pixel_diff

# lsb_xor_algorithm.py
from lsb_xor_algorithm import (
    select_complex_indices_from_image,
    embed_xor_lsb_at_indices,
    extract_xor_lsb_at_indices,
    flat_to_image,
    image_to_flat
)

# ------------------------------------------------------

app = Flask(__name__)
toastr = Toastr(app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

# =====================================================
# =============== WAV / PCM AUDIO HELPERS =============
# =====================================================

def _rng_from_key(key: str) -> np.random.RandomState:
    # Make a stable seed from the key string
    seed = np.frombuffer(key.encode("utf-8"), dtype=np.uint8).sum(dtype=np.uint32)
    return np.random.RandomState(int(seed) & 0x7FFFFFFF)

def _wav_bytes_to_np(wav_bytes: bytes):
    """
    Return (samples_1d, params_dict).
    samples_1d is a 1-D numpy array of dtype:
        uint8 for 8-bit PCM
        int16 for 16-bit PCM
        int32 for 24-bit PCM (stored in 32 bits)
    Interleaved channels are preserved in the 1-D order.
    """
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()  # bytes per sample (1,2,3)
        framerate = w.getframerate()
        n_frames = w.getnframes()
        comptype = w.getcomptype()
        compname = w.getcompname()
        frames = w.readframes(n_frames)

    if comptype != "NONE":
        raise ValueError("Compressed WAV not supported. Use PCM (uncompressed).")

    if sampwidth == 1:
        samples = np.frombuffer(frames, dtype=np.uint8).copy()
        dtype = np.uint8
    elif sampwidth == 2:
        samples = np.frombuffer(frames, dtype="<i2").copy()
        dtype = np.int16
    elif sampwidth == 3:
        b = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        s = (b[:, 0].astype(np.uint32)
             | (b[:, 1].astype(np.uint32) << 8)
             | (b[:, 2].astype(np.uint32) << 16))
        sign = (s & 0x800000) != 0
        s_signed = s.astype(np.int32)
        s_signed[sign] -= (1 << 24)
        samples = s_signed
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth*8} bits")

    params = dict(
        n_channels=n_channels,
        sampwidth=sampwidth,
        framerate=framerate,
        n_frames=n_frames,
        dtype=dtype,
    )
    return samples, params

def _np_to_wav_bytes(samples: np.ndarray, params: dict) -> bytes:
    """
    Convert 1-D interleaved samples back into WAV bytes with original params.
    """
    n_channels = params["n_channels"]
    sampwidth = params["sampwidth"]
    framerate = params["framerate"]

    samples = np.ascontiguousarray(samples)

    if sampwidth == 1:
        if samples.dtype != np.uint8:
            samples = np.clip(samples, 0, 255).astype(np.uint8)
        frames_bytes = samples.tobytes()
    elif sampwidth == 2:
        if samples.dtype != np.int16:
            samples = np.clip(samples, -32768, 32767).astype("<i2")
        frames_bytes = samples.astype("<i2").tobytes()
    elif sampwidth == 3:
        s = samples.astype(np.int32)
        s = np.clip(s, -(1 << 23), (1 << 23) - 1)
        s_u = s.copy()
        neg = s_u < 0
        s_u[neg] += (1 << 24)
        b0 = (s_u & 0xFF).astype(np.uint8)
        b1 = ((s_u >> 8) & 0xFF).astype(np.uint8)
        b2 = ((s_u >> 16) & 0xFF).astype(np.uint8)
        frames_bytes = np.column_stack([b0, b1, b2]).ravel().tobytes()
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth*8} bits")

    bio_out = io.BytesIO()
    with wave.open(bio_out, "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(frames_bytes)
    return bio_out.getvalue()

def _select_audio_indices(samples: np.ndarray, key: str, top_percent: int | None = None) -> np.ndarray:
    """
    For now: use ALL sample positions, shuffled by key.
    (You can later swap to an energy-based selection and keep the same shape.)
    """
    n = samples.size
    idx = np.arange(n, dtype=np.int64)
    rng = _rng_from_key(key)
    rng.shuffle(idx)
    return idx

def _embed_audio_wav(wav_bytes: bytes, payload: bytes, k: int, key: str) -> bytes:
    samples, params = _wav_bytes_to_np(wav_bytes)
    # Choose indices (all samples shuffled for now)
    indices = _select_audio_indices(samples, key)

    bits_needed = (16 + len(payload)) * 8  # header + payload (your engine uses 16-byte header)
    bits_avail = indices.size * k
    if bits_needed > bits_avail:
        raise ValueError(
            f"Not enough capacity in selected audio samples. Need {bits_needed} bits, have {bits_avail} bits."
        )

    # IMPORTANT: operate on a copy to avoid mutating input
    work = samples.copy()

    # Reuse your robust header + keyed XOR + LSB function
    stego = embed_xor_lsb_at_indices(work, payload, k=k, key=key, indices=indices)
    
    # Pack back to WAV
    return _np_to_wav_bytes(stego, params)

def _extract_audio_wav(wav_bytes: bytes, k: int, key: str) -> bytes:
    samples, params = _wav_bytes_to_np(wav_bytes)
    indices = _select_audio_indices(samples, key)
    payload = extract_xor_lsb_at_indices(samples, k=k, key=key, indices=indices)
    return payload

# =====================================================
# ==================== ROUTES =========================
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():
    # If the main form posts to "/", delegate to the embed logic.
    if request.method == "POST":
        return embed_media()
    return render_template("index.html")

@app.route("/results")
def results():
    cover_filepath = session.get("cover_image")
    stego_filepath = session.get("stego_image")

    if not cover_filepath or not stego_filepath:
        flash("No current files found. Please embed your file first.", "error")
        return redirect(url_for("index"))

    if cover_filepath.lower().endswith(".png") and stego_filepath.lower().endswith(".png"):
        media_type = "img"
    elif cover_filepath.lower().endswith(".wav") and stego_filepath.lower().endswith(".wav"):
        media_type = "audio"
    else:
        flash("File incompatible. Please embed your file first.", "error")
        return redirect(url_for("index"))

    difference = compute_pixel_diff(cover_filepath, stego_filepath)
    return render_template(
        "results.html",
        cover_filepath=cover_filepath,
        stego_filepath=stego_filepath,
        media_type=media_type,
        difference=difference,
    )

@app.route("/check", methods=["POST"])
def check_capacity_form():
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
            flash(f"Audio capacity: {cap} bytes available (theoretical).", "success")
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
def embed_media():
    """
    Embed payload into an IMAGE (PNG/BMP/GIF) or AUDIO (WAV) using LSB+XOR engine.
    """
    cover = request.files.get("coverFile")
    lsb_str = request.form.get("lsbCount", "1")
    key = request.form.get("stegoKey", "").strip()
    text_payload_str = request.form.get("textPayload", "")
    payload_file = next((f for f in request.files.getlist("payloadFile") if f and f.filename), None)

    if not cover or not cover.filename:
        flash("Upload a cover file first.", "error")
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
    cover_bytes = cover.read()

    # -------- IMAGE PATH --------
    if is_image_extension(cover.filename):
        try:
            meta_img = load_image_meta(cover_bytes)
            capacity_bytes = compute_capacity_bytes_image(meta_img, lsb)
            if len(payload_bytes) > capacity_bytes:
                flash(f"Payload too large: {len(payload_bytes)} > {capacity_bytes} bytes (theoretical).", "error")
                return redirect(url_for("index"))

            flat_cover, shape, _ = image_to_flat(cover_bytes, mode="RGB")
            _, _, _, eligible = select_complex_indices_from_image(
                cover_bytes, top_percent=30, mode="RGB", key=key
            )
            bits_needed = (16 + len(payload_bytes)) * 8
            bits_avail = len(eligible) * lsb
            if bits_needed > bits_avail:
                flash(
                    "Not enough complex locations with current LSB/top% settings. "
                    f"Need {bits_needed} bits, have {bits_avail} bits.",
                    "error"
                )
                return redirect(url_for("index"))

            stego_flat = embed_xor_lsb_at_indices(
                flat_cover, payload_bytes, k=lsb, key=key, indices=eligible
            )
            stego_png = flat_to_image(stego_flat, shape, mode="RGB")

            # Best-effort session save; don't fail the request if it can't be saved.
            try:
                save_images_to_session(cover_bytes, stego_png)
            except Exception:
                pass

            return send_file(
                io.BytesIO(stego_png),
                mimetype="image/png",
                as_attachment=True,
                download_name="stego.png",
            )
        except Exception as e:
            flash(f"Embed (image) failed: {e}", "error")
            return redirect(url_for("index"))

    # -------- AUDIO PATH (WAV) --------
    if is_wav_extension(cover.filename):
        try:
            meta = load_audio_meta(cover_bytes)
            if not meta.is_pcm_integer:
                flash("WAV must be 8/16/24-bit PCM.", "error")
                return redirect(url_for("index"))

            capacity_bytes = compute_capacity_bytes(meta, lsb)
            if len(payload_bytes) > capacity_bytes:
                flash(f"Payload too large: {len(payload_bytes)} > {capacity_bytes} bytes (theoretical).", "error")
                return redirect(url_for("index"))

            stego_wav = _embed_audio_wav(cover_bytes, payload_bytes, k=lsb, key=key)
            return send_file(
                io.BytesIO(stego_wav),
                mimetype="audio/wav",
                as_attachment=True,
                download_name="stego.wav",
            )
        except Exception as e:
            flash(f"Embed (audio) failed: {e}", "error")
            return redirect(url_for("index"))

    flash("Unsupported cover type. Use PNG/BMP/GIF for images or WAV for audio.", "error")
    return redirect(url_for("index"))

@app.route("/extract", methods=["POST"])
def extract_media():
    """
    Extract payload from IMAGE (PNG/BMP/GIF) or AUDIO (WAV) using the same LSB/key.
    """
    stego = request.files.get("stegoFile")
    lsb_str = request.form.get("lsbCount", "1")
    key = request.form.get("stegoKey", "").strip()

    if not stego or not stego.filename:
        flash("Upload a stego file.", "error")
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

    file_bytes = stego.read()

    # -------- IMAGE PATH --------
    if is_image_extension(stego.filename):
        try:
            flat, _, _ = image_to_flat(file_bytes, mode="RGB")
            _, _, _, eligible = select_complex_indices_from_image(
                file_bytes, top_percent=30, mode="RGB", key=key
            )
            payload = extract_xor_lsb_at_indices(flat, k=lsb, key=key, indices=eligible)
            return send_file(
                io.BytesIO(payload),
                mimetype="application/octet-stream",
                as_attachment=True,
                download_name="extracted_payload.bin",
            )
        except Exception as e:
            flash(f"Extract (image) failed: {e}", "error")
            return redirect(url_for("index"))

    # -------- AUDIO PATH --------
    if is_wav_extension(stego.filename):
        try:
            payload = _extract_audio_wav(file_bytes, k=lsb, key=key)
            return send_file(
                io.BytesIO(payload),
                mimetype="application/octet-stream",
                as_attachment=True,
                download_name="extracted_payload.bin",
            )
        except Exception as e:
            flash(f"Extract (audio) failed: {e}", "error")
            return redirect(url_for("index"))

    flash("Unsupported file type for extraction.", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.debug = True
    app.run()
