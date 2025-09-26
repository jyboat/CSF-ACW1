from flask_toastr import Toastr
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify, session
)
import io, logging
import os
import wave
import numpy as np
from PIL import Image
# from zipfile import ZipFile, ZIP_DEFLATED

# capacity.py
from capacity import (
    is_wav_extension,
    is_image_extension,
    is_gif_extension,
    load_audio_meta,
    load_image_meta,
    compute_capacity_bytes,
    compute_capacity_bytes_image,
)

# data_comparison helpers
from data_comparison import (
    detect_has_alpha,
    save_images_to_session,
    save_audio_to_session,
    compute_pixel_diff,
    compute_audio_diff,
    save_rgb_analysis_to_session,
    save_audio_analysis_to_session,
    save_image_comparison_to_session,
    save_gray_analysis_to_session,
    save_spectrogram_comparison_to_session,
    save_gif_diff_animation_to_session
    )

# lsb_xor_algorithm.py  
from lsb_xor_algorithm import (
    image_to_flat,
    flat_to_image,
    embed_xor_lsb_from_xy,              
    embed_xor_lsb_auto,                 
    extract_xor_lsb_auto,               
    embed_xor_lsb_audio, extract_xor_lsb_audio,
    embed_uuid_box_mp4,
    extract_uuid_box_mp4
)

from lsb_xor_gif_multiframe import (
    compute_gif_multiframe_capacity,
    embed_gif_multiframe_lsb_xor,
    extract_gif_multiframe_lsb_xor
)


from capacity import is_mp4_extension, load_mp4_meta, compute_capacity_bytes_mp4
from data_comparison import save_video_to_session

# ------------------------------------------------------

app = Flask(__name__)
toastr = Toastr(app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

# Make sure INFO shows up in your terminal
app.logger.setLevel(logging.INFO)
if not app.logger.handlers:
    app.logger.addHandler(logging.StreamHandler())

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

def _select_audio_indices(samples: np.ndarray, key: str) -> np.ndarray:
    """
    For now: use ALL sample positions, shuffled by key.
    (You can later swap to an energy-based selection and keep the same shape.)
    """
    n = samples.size
    idx = np.arange(n, dtype=np.int64)
    rng = _rng_from_key(key)
    rng.shuffle(idx)
    return idx

def _embed_audio_wav(wav_bytes: bytes, payload: bytes, k: int, key: str,
                     start_sample: int = 0, use_complex: bool = False) -> bytes:
    samples, params = _wav_bytes_to_np(wav_bytes)
    # Choose indices (all samples shuffled for now)
    indices = _select_audio_indices(samples, key)

    bits_needed = (32 + len(payload)) * 8  # header + payload (your engine uses 16-byte header)
    bits_avail = indices.size * k
    if bits_needed > bits_avail:
        raise ValueError(
            f"Not enough capacity in selected audio samples. Need {bits_needed} bits, have {bits_avail} bits."
        )

    # IMPORTANT: operate on a copy to avoid mutating input
    work = samples.copy()

    # Reuse your robust header + keyed XOR + LSB function
    stego = embed_xor_lsb_audio(work, payload, k=k, key=key, start_sample=start_sample,
                                   use_complex=use_complex)
    
    # Pack back to WAV
    return _np_to_wav_bytes(stego, params)

def _extract_audio_wav(wav_bytes: bytes, k: int, key: str) -> bytes:
    samples, _ = _wav_bytes_to_np(wav_bytes)
    payload, _hdr = extract_xor_lsb_audio(samples, k=k, key=key)
    return payload

# ====================================================
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
    cover_filepath = session.get("cover")
    stego_filepath = session.get("stego")

    if not cover_filepath or not stego_filepath:
        flash("No current files found. Please embed your file first.", "error")
        return redirect(url_for("index"))

    # IMAGE / GIF path
    valid_extensions = (".png", ".bmp", ".gif")
    if cover_filepath.lower().endswith(valid_extensions) and stego_filepath.lower().endswith(valid_extensions):
        media_type = "img"

        # Basic pixel diff (for quick counts/coords; for GIFs this is effectively first frame)
        difference = compute_pixel_diff(cover_filepath, stego_filepath)

        # Standard image analyses
        save_rgb_analysis_to_session(cover_filepath, stego_filepath)
        rgb_analysis = session.get("rgb_analysis_filepath")

        save_image_comparison_to_session(cover_filepath, stego_filepath)
        diff_visual = session.get("diff_visualization_filepath")

        save_gray_analysis_to_session(cover_filepath, stego_filepath)
        gray_analysis = session.get("gray_analysis_filepath")

        # --- Animated GIF specific: build an animated per-frame diff if possible ---
        gif_diff_filepath = None
        if cover_filepath.lower().endswith(".gif") and stego_filepath.lower().endswith(".gif"):
            try:

                with Image.open("static/" + cover_filepath) as c, Image.open("static/" + stego_filepath) as s:
                    if getattr(c, "is_animated", False) and getattr(s, "is_animated", False):
                        save_gif_diff_animation_to_session(cover_filepath, stego_filepath)
                        gif_diff_filepath = session.get("gif_diff_filepath")
            except Exception as e:
                # Non-fatal: we still render the page without the animated diff
                app.logger.warning(f"GIF animated diff could not be generated: {e}")

        return render_template(
            "results.html",
            media_type=media_type,
            cover_filepath=cover_filepath,
            stego_filepath=stego_filepath,
            difference=difference,
            rgb_analysis_filepath=rgb_analysis,
            diff_visualization_filepath=diff_visual,
            gray_analysis_filepath=gray_analysis,
            gif_diff_filepath=gif_diff_filepath  # may be None if not animated or on error
        )

    # AUDIO path
    elif cover_filepath.lower().endswith(".wav") and stego_filepath.lower().endswith(".wav"):
        media_type = "audio"
        difference = compute_audio_diff(cover_filepath, stego_filepath)

        cover_audiopath = cover_filepath
        stego_audiopath = stego_filepath

        save_audio_analysis_to_session(cover_filepath, stego_filepath)
        audio_analysis = session.get("audio_analysis_filepath")

        save_spectrogram_comparison_to_session(cover_filepath, stego_filepath)
        audio_spectrogram = session.get("audio_spectrogram_filepath")

        return render_template(
            "results.html",
            media_type=media_type,
            cover_audiopath=cover_audiopath,
            stego_audiopath=stego_audiopath,
            difference=difference,
            audio_analysis_filepath=audio_analysis,
            audio_spectrogram_filepath=audio_spectrogram,
        )

    # VIDEO (MP4) path
    elif cover_filepath.lower().endswith(".mp4") and stego_filepath.lower().endswith(".mp4"):
        media_type = "mp4"
        cover_videopath = cover_filepath
        stego_videopath = stego_filepath


        return render_template(
            "results.html",
            media_type=media_type,
            cover_videopath=cover_videopath,
            stego_videopath=stego_videopath,
        )

    else:
        flash("File incompatible. Please embed your file first.", "error")
        return redirect(url_for("index"))

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
        elif is_mp4_extension(cover.filename):
            meta = load_mp4_meta(cover_bytes)
            cap = compute_capacity_bytes_mp4(meta)
        elif is_gif_extension(cover.filename):
            cap = compute_gif_multiframe_capacity(cover_bytes, lsb)  # ALL frames
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
    user_key = request.form.get("stegoKey", "").strip()

    start_x_str = request.form.get("startX", "").strip()
    start_y_str = request.form.get("startY", "").strip()
    key = user_key

    app.logger.info("EMBED fields → useComplex=%r startSample=%r lsb=%r key=%r",
                    request.form.get("useComplex"),
                    request.form.get("startSample"),
                    request.form.get("lsbCount"),
                    request.form.get("stegoKey"))

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

    text_payload_str = request.form.get("textPayload", "")
    payload_file = next((f for f in request.files.getlist("payloadFile") if f and f.filename), None)

    # Accept either text or file payload
    if text_payload_str and payload_file:
        flash("Choose text OR a file (not both).", "error")
        return redirect(url_for("index"))
    if not text_payload_str and not payload_file:
        flash("Provide a payload (text or file).", "error")
        return redirect(url_for("index"))

    payload_bytes = text_payload_str.encode("utf-8") if text_payload_str else payload_file.read()
    cover_bytes = cover.read()

    # -------- ANIMATED GIF PATH --------
    if is_gif_extension(cover.filename):
        try:
            # Calculate capacity across ALL frames
            capacity_bytes = compute_gif_multiframe_capacity(cover_bytes, lsb)
            if len(payload_bytes) > capacity_bytes:
                flash(f"Payload too large: {len(payload_bytes)} > {capacity_bytes} bytes (across all frames).", "error")
                return redirect(url_for("index"))

            app.logger.info(f"Embedding {len(payload_bytes)} bytes across all GIF frames (capacity: {capacity_bytes})")

            # Embed across ALL frames using LSB+XOR algorithm
            # if start_x_str and start_y_str:
            #     stego_gif = embed_gif_multiframe_lsb_xor(cover_bytes, payload_bytes, k=lsb, key=key, click_x=int(start_x_str), click_y=int(start_y_str))
            # else:
            stego_gif = embed_gif_multiframe_lsb_xor(cover_bytes, payload_bytes, k=lsb, key=key)
            
            save_images_to_session(cover_bytes, stego_gif, img_format="gif")
            return redirect(url_for("results"))

        except Exception as e:
            flash(f"Embed (multi-frame GIF) failed: {e}", "error")
            return redirect(url_for("index"))
    
    # -------- IMAGE PATH --------
    if is_image_extension(cover.filename):
        try:
            ext = os.path.splitext(cover.filename)[1].lower()
            img_format = ext.lstrip('.').upper()               # "PNG"

            meta_img = load_image_meta(cover_bytes)
            capacity_bytes = compute_capacity_bytes_image(meta_img, lsb)
            if len(payload_bytes) > capacity_bytes:
                flash(f"Payload too large: {len(payload_bytes)} > {capacity_bytes} bytes (theoretical).", "error")
                return redirect(url_for("index"))

            has_alpha = detect_has_alpha(cover_bytes)
            mode = "RGBA" if has_alpha else "RGB"



            flat_cover, shape, _ = image_to_flat(cover_bytes, mode=mode)

            if start_x_str and start_y_str:
                # user clicked; use their (x,y)
                sx = int(start_x_str); sy = int(start_y_str)
                stego_flat = embed_xor_lsb_from_xy(
                    flat_cover, shape, payload_bytes, k=lsb, key=key,
                    start_x=sx, start_y=sy
                )
            else:
                # auto-pick complex start (keyed) + embed
                stego_flat = embed_xor_lsb_auto(
                    cover_bytes, flat_cover, shape, payload_bytes, k=lsb, key=key, mode=mode
                )

            stego_png = flat_to_image(stego_flat, shape, mode=mode, img_format=img_format)


            save_images_to_session(cover_bytes, stego_png, img_format=img_format)

            # Build ZIP in memory
            # zip_buf = io.BytesIO()
            # with ZipFile(zip_buf, "w", ZIP_DEFLATED) as zf:
            #     zf.writestr("stego.png", stego_png)
            #     zf.writestr("readme.txt", "Extraction: upload stego image, enter same LSB and key. (start_x,start_y) are stored in the image)")

            # GENERATED_DIR = os.path.join(os.getcwd(), "zip")
            # os.makedirs(GENERATED_DIR, exist_ok=True)

            # bundle_name = f"stego_bundle_{uuid.uuid4().hex}.zip"
            # bundle_path = os.path.join(GENERATED_DIR, bundle_name)
            # with open(bundle_path, "wb") as f:
            #     f.write(zip_buf.getvalue())

            # session["stego_bundle"] = bundle_path
            return redirect(url_for("results"))

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

            start_sample_str = request.form.get("startSample", "").strip()
            auto_flag = (request.form.get("autoStart") or request.form.get("useComplex") or "").strip().lower()
            use_complex = auto_flag in {"1", "true"}
            start_sample = int(start_sample_str) if start_sample_str.isdigit() else 0

            stego_wav = _embed_audio_wav(
                cover_bytes, payload_bytes, k=lsb, key=key,
                start_sample=start_sample,
                use_complex=use_complex
            )

            save_audio_to_session(cover_bytes, stego_wav)
            return redirect(url_for("results"))

        except Exception as e:
            flash(f"Embed (audio) failed: {e}", "error")
            return redirect(url_for("index"))
    
    # -------- MP4 PATH --------
    if is_mp4_extension(cover.filename):
        try:
            meta = load_mp4_meta(cover_bytes)
            capacity_bytes = compute_capacity_bytes_mp4(meta)

            if len(payload_bytes) > capacity_bytes:
                flash(f"Payload too large: {len(payload_bytes)} > {capacity_bytes} bytes.", "error")
                return redirect(url_for("index"))

            stego_bytes = embed_uuid_box_mp4(cover_bytes, payload_bytes, k=lsb, key=key)

            save_video_to_session(cover_bytes, stego_bytes)
            return redirect(url_for("results"))
        except Exception as e:
            flash(f"Embed (mp4) failed: {e}", "error")
            return redirect(url_for("index"))

    flash("Unsupported cover type. Use PNG/BMP/GIF for images or WAV for audio.", "error")
    app.logger.info(
        "EMBED fields → useComplex=%r startSample=%r lsb=%r key=%r",
        request.form.get("useComplex"),
        request.form.get("startSample"),
        request.form.get("lsbCount"),
        request.form.get("stegoKey"),
    )
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
    if is_gif_extension(stego.filename):
        try:
            app.logger.info("Extracting payload from all GIF frames")
            
            # Extract from ALL frames using your LSB+XOR algorithm
            payload = extract_gif_multiframe_lsb_xor(file_bytes, k=lsb, key=key)
            
            return send_file(
                io.BytesIO(payload),
                mimetype="application/octet-stream",
                as_attachment=True, 
                download_name="extracted_payload.bin",
            )
        except Exception as e:
            flash(f"Extract (multi-frame GIF) failed: {e}", "error")
            return redirect(url_for("index"))
            
    if is_image_extension(stego.filename):
        try:
            has_alpha = detect_has_alpha(file_bytes)
            mode = "RGBA" if has_alpha else "RGB"

            flat, shape, _ = image_to_flat(file_bytes, mode=mode)

            # Reads STG2 header at (0,0), then payload from (x,y) stored in header
            payload = extract_xor_lsb_auto(flat, shape, k=lsb, key=key)

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
        
    # -------- MP4 PATH -------- 
    if is_mp4_extension(stego.filename):
        try:
            payload = extract_uuid_box_mp4(file_bytes, k=lsb, key=key)
            return send_file(
                io.BytesIO(payload),
                mimetype="application/octet-stream",
                as_attachment=True,
                download_name="payload.bin"
            )
        except Exception as e:
            flash(f"Extract (mp4) failed: {e}", "error")
            return redirect(url_for("index"))

    flash("Unsupported file type for extraction.", "error")
    return redirect(url_for("index"))

@app.route("/download/bundle")
def download_bundle():
    bundle_path = session.get("stego_bundle")
    if not bundle_path or not os.path.exists(bundle_path):
        flash("No bundle available. Please embed again.", "error")
        return redirect(url_for("index"))
    return send_file(
        bundle_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=os.path.basename(bundle_path),
    )

@app.route("/hexcompare")
def hexcompare():
    cover_path = session.get("cover")
    stego_path = session.get("stego")

    if not cover_path or not stego_path:
        flash("No current files found. Please embed again.", "error")
        return redirect(url_for("index"))

    if cover_path.lower().endswith(".png") and stego_path.lower().endswith(".png"):
        media_type = "img"
        return render_template(
            "hexcompare.html",
            media_type=media_type,
            cover_filepath=cover_path,
            stego_filepath=stego_path,
        )

    if cover_path.lower().endswith(".wav") and stego_path.lower().endswith(".wav"):
        media_type = "audio"
        return render_template(
            "hexcompare.html",
            media_type=media_type,
            cover_audiopath=cover_path,
            stego_audiopath=stego_path,
        )
        
    if cover_path.lower().endswith(".mp4") and stego_path.lower().endswith(".mp4"):
        media_type = "mp4"
        return render_template(
            "hexcompare.html",
            media_type=media_type,
            cover_videopath=cover_path,
            stego_videopath=stego_path,
        )


    flash("Unsupported media for hex compare.", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.debug = True
    app.run()
