import io, struct as _struct, hashlib, random, uuid
from dataclasses import dataclass
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Dataclass for audios
@dataclass
class AudioHeader:
    magic: bytes = b"STG2"
    length: int = 0
    sha8: bytes = b""
    start_sample: int = 0
    flags: int = 0
    _pad: int = 0

# =========================
# ===== IMAGE HELPERS =====
# =========================

def image_to_flat(image_bytes: bytes, mode: str = "RGB"):
    im = Image.open(io.BytesIO(image_bytes)).convert(mode)
    arr = np.array(im, dtype=np.uint8)        # (H, W, C) or (H, W)
    return arr.reshape(-1), arr.shape, im.mode

def flat_to_image(flat: np.ndarray, shape: tuple, mode: str = "RGB", img_format="PNG") -> bytes:
    arr = flat.reshape(shape).astype(np.uint8)
    im = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    im.save(buf, format=img_format)                # lossless
    buf.seek(0)
    return buf.getvalue()

# ============================
# ====== KEYSTREAM (XOR) =====
# ============================

def keystream_bits(key: str, n_bits: int) -> np.ndarray:
    if n_bits <= 0:
        return np.empty(0, dtype=np.uint8)
    counter, out = 0, []
    key_bytes = key.encode()
    while len(out) < n_bits:
        block = hashlib.sha256(key_bytes + _struct.pack(">Q", counter)).digest()
        for byte in block:
            for i in range(8):
                out.append((byte >> (7 - i)) & 1)
                if len(out) >= n_bits:
                    return np.array(out, dtype=np.uint8)
        counter += 1

# =========================================
# ===== IMAGE HEADER (STG2 with x,y)  =====
# =========================================

@dataclass
class Header:
    magic: bytes = b"STG2"      # 4
    length: int = 0             # 4 (uint32 LE)
    sha8: bytes = b""           # 8
    start_x: int = 0            # 4 (int32 LE)
    start_y: int = 0            # 4 (int32 LE)

IMG_HDR_V2_SIZE = 24  # total bytes

def make_header(payload: bytes, start_x: int = 0, start_y: int = 0) -> bytes:
    h = Header(
        length=len(payload),
        sha8=hashlib.sha256(payload).digest()[:8],
        start_x=int(start_x),
        start_y=int(start_y),
    )
    return (
        h.magic +
        _struct.pack("<I", h.length) +
        h.sha8 +
        _struct.pack("<i", h.start_x) +
        _struct.pack("<i", h.start_y)
    )

def parse_header(hdr: bytes) -> Header:
    if len(hdr) < IMG_HDR_V2_SIZE or hdr[:4] != b"STG2":
        raise ValueError("Invalid STG2 image header")
    length  = _struct.unpack("<I", hdr[4:8])[0]
    sha8    = hdr[8:16]
    start_x = _struct.unpack("<i", hdr[16:20])[0]
    start_y = _struct.unpack("<i", hdr[20:24])[0]
    return Header(magic=b"STG2", length=length, sha8=sha8, start_x=start_x, start_y=start_y)

# =====================================
# ===== LINEAR INDEXING FROM (x,y) =====
# =====================================

def _linear_indices_from_xy(shape: tuple, start_x: int, start_y: int) -> np.ndarray:
    # Gray Scale (H, W)
    if len(shape) == 2:
        H, W = shape
        C = 1
    # Colored image (H, W, C)
    elif len(shape) == 3:
        H, W, C = shape
    else:
        raise ValueError(f"Unsupported image shape: {shape}")

    # Check bounds
    if not (0 <= int(start_x) < W and 0 <= int(start_y) < H):
        raise ValueError(f"start_x/start_y out of bounds for image {W}x{H}")

    start_idx = (int(start_y) * W + int(start_x)) * C  # start at channel 0 of (x,y)
    N = H * W * C

    idx = np.arange(N, dtype=np.int64)
    # rotate so we start at (x,y,*)
    if start_idx > 0:
        idx = np.concatenate([idx[start_idx:], idx[:start_idx]])
    return idx

def _indices_excluding_alpha(shape: tuple, indices: np.ndarray, flat_cover = None) -> np.ndarray:
    """
    - Skip alpha channel in RGBA.
    - If RGBA *and* flat_cover is provided, also skip RGB of pixels where alpha==0.
    """
    if len(shape) == 3 and shape[2] == 4:
        # drop alpha channel positions
        keep = (indices % 4 != 3)
        indices = indices[keep]
        if flat_cover is not None:
            # alpha bytes are every 4th byte starting at offset 3
            alpha = flat_cover.reshape(-1)[3::4]  # length = H*W
            # pixel index for each remaining channel byte
            pix = indices // 4
            visible = (alpha[pix] != 0)
            indices = indices[visible]
        return indices
    return indices

def _write_bits_lsb_at_indices(target: np.ndarray, bits: np.ndarray, k: int, indices: np.ndarray) -> None:
    pad = (-len(bits)) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    groups = bits.reshape(-1, k)
    if groups.shape[0] > len(indices):
        raise ValueError("Not enough indices to embed bits")
    mask = ~((1 << k) - 1)
    for i, grp in enumerate(groups):
        v = 0
        for b in grp:
            v = (v << 1) | int(b)
        j = int(indices[i])
        target[j] = (int(target[j]) & mask) | v

def _read_bits_lsb_at_indices(source: np.ndarray, k: int, n_bits: int, indices: np.ndarray) -> np.ndarray:
    groups = (n_bits + k - 1) // k
    if groups > len(indices):
        raise ValueError("Not enough indices to read bits")
    out = []
    for i in range(groups):
        v = int(source[int(indices[i])]) & ((1 << k) - 1)
        for t in range(k - 1, -1, -1):
            out.append((v >> t) & 1)
    return np.array(out[:n_bits], dtype=np.uint8)

# =====================================
# ====== COMPLEXITY-BASED START XY =====
# =====================================

def _sobel_magnitude(gray_uint8: np.ndarray) -> np.ndarray:
    g = gray_uint8.astype(np.float32)
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    gpad = np.pad(g, 1, mode='reflect')
    Gx = (kx[0,0]*gpad[:-2,:-2] + kx[0,1]*gpad[:-2,1:-1] + kx[0,2]*gpad[:-2,2:] +
          kx[1,0]*gpad[1:-1,:-2] + kx[1,1]*gpad[1:-1,1:-1] + kx[1,2]*gpad[1:-1,2:] +
          kx[2,0]*gpad[2:,  :-2] + kx[2,1]*gpad[2:,  1:-1] + kx[2,2]*gpad[2:,  2:])
    Gy = (ky[0,0]*gpad[:-2,:-2] + ky[0,1]*gpad[:-2,1:-1] + ky[0,2]*gpad[:-2,2:] +
          ky[1,0]*gpad[1:-1,:-2] + ky[1,1]*gpad[1:-1,1:-1] + ky[1,2]*gpad[1:-1,2:] +
          ky[2,0]*gpad[2:,  :-2] + ky[2,1]*gpad[2:,  1:-1] + ky[2,2]*gpad[2:,  2:])
    return np.hypot(Gx, Gy)

def _auto_start_xy(image_bytes: bytes, mode: str, key: str, top_percent: float = 0.5) -> tuple[int, int]:
    """
    Pick a high-complexity *visible* pixel.
    - Load as RGBA when available.
    - Compute Sobel on gray, weighted by alpha (so transparent areas score ~0).
    - Restrict the candidate pool to alpha>0 pixels.
    """
    im = Image.open(io.BytesIO(image_bytes))
    if "A" in (mode or ""):
        im = im.convert("RGBA")
    else:
        im = im.convert("RGB")

    arr = np.array(im, dtype=np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[..., :3].astype(np.float32)
        a   = arr[..., 3].astype(np.float32) / 255.0
        # luminance
        gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2])
        # damp gray where alpha is low so edges in transparent regions are deprioritized
        gray = (gray * a).astype(np.uint8)
        mag  = _sobel_magnitude(gray)
        visible = (a > 0.0)
    else:
        gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.uint8) if arr.ndim == 3 else arr
        mag  = _sobel_magnitude(gray)
        visible = np.ones_like(mag, dtype=bool)

    H, W = mag.shape
    # consider only visible pixels
    vis_idx = np.flatnonzero(visible.ravel())
    if vis_idx.size == 0:
        # fallback: no alpha or everything invisible; use all pixels
        vis_idx = np.arange(H*W, dtype=np.int64)

    # take the top X% (default: 0.5% as before)
    k = max(1, int(vis_idx.size * (top_percent / 100.0)))
    # magnitudes only over visible indices
    vis_mag = mag.ravel()[vis_idx]
    top_k_idx_in_vis = np.argpartition(vis_mag, -k)[-k:]
    pool = vis_idx[top_k_idx_in_vis]

    # deterministic keyed pick
    seed_material = f"{key}|{H}|{W}|{arr.shape[2] if arr.ndim==3 else 1}".encode()
    pick = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], 'little') % pool.size
    chosen = int(pool[pick])
    y, x = divmod(chosen, W)
    return x, y

# =====================================
# ======= EMBED / EXTRACT (IMAGE) =====
# =====================================

def embed_xor_lsb_from_xy(cover_flat: np.ndarray, shape: tuple,
                          payload: bytes, k: int, key: str,
                          start_x: int, start_y: int) -> np.ndarray:
    """
    Writes STG2 header @ (0,0), then writes payload starting at (start_x, start_y)
    (wrapping), skipping alpha channel positions.
    """
    out = cover_flat.copy()

    # 1) HEADER @ (0,0)
    hdr_bytes = make_header(payload, start_x=start_x, start_y=start_y)
    H_bits = np.unpackbits(np.frombuffer(hdr_bytes, dtype=np.uint8))
    K_hdr  = keystream_bits(key, len(H_bits))
    C_hdr  = H_bits ^ K_hdr

    # header indices
    idx_hdr = _indices_excluding_alpha(shape, _linear_indices_from_xy(shape, 0, 0), cover_flat)

    _write_bits_lsb_at_indices(out, C_hdr, k, idx_hdr)

    # 2) PAYLOAD @ (start_x, start_y)
    P_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    K_pay  = keystream_bits(key, len(P_bits))
    C_pay  = P_bits ^ K_pay

    # payload indices
    idx_pay = _indices_excluding_alpha(shape, _linear_indices_from_xy(shape, start_x, start_y), cover_flat)

    # Avoid overlap if start is (0,0): skip header groups
    header_groups = (IMG_HDR_V2_SIZE * 8 + k - 1) // k
    if start_x == 0 and start_y == 0:
        idx_pay = idx_pay[header_groups:]

    _write_bits_lsb_at_indices(out, C_pay, k, idx_pay)
    return out

def embed_xor_lsb_auto(cover_bytes: bytes, cover_flat: np.ndarray, shape: tuple,
                       payload: bytes, k: int, key: str, mode: str) -> np.ndarray:
    sx, sy = _auto_start_xy(cover_bytes, mode=mode, key=key, top_percent=0.5)
    return embed_xor_lsb_from_xy(cover_flat, shape, payload, k, key, sx, sy)

def extract_xor_lsb_auto(stego_flat: np.ndarray, shape: tuple, k: int, key: str) -> bytes:
    """
    Read STG2 header at (0,0) to get length + (start_x, start_y), then read payload from there.
    """
    # 1) header from (0,0)
    idx_hdr = _indices_excluding_alpha(shape, _linear_indices_from_xy(shape, 0, 0), stego_flat)
    C_hdr   = _read_bits_lsb_at_indices(stego_flat, k, IMG_HDR_V2_SIZE * 8, idx_hdr)
    K_hdr   = keystream_bits(key, len(C_hdr))
    M_hdr   = C_hdr ^ K_hdr
    hdr_bytes = np.packbits(M_hdr).tobytes()
    hdr = parse_header(hdr_bytes)

    # 2) payload from (start_x,start_y)
    total_bits = hdr.length * 8
    idx_pay = _indices_excluding_alpha(shape, _linear_indices_from_xy(shape, hdr.start_x, hdr.start_y), stego_flat)

    # If payload started at (0,0), skip the header region
    header_groups = (IMG_HDR_V2_SIZE * 8 + k - 1) // k
    if hdr.start_x == 0 and hdr.start_y == 0:
        idx_pay = idx_pay[header_groups:]

    C_pay = _read_bits_lsb_at_indices(stego_flat, k, total_bits, idx_pay)
    K_pay = keystream_bits(key, len(C_pay))
    M_pay = C_pay ^ K_pay
    payload = np.packbits(M_pay).tobytes()

    if hashlib.sha256(payload).digest()[:8] != hdr.sha8:
        raise ValueError("Checksum mismatch")
    return payload

# Run immediately

# cover_path = "test/img_rgb_100x100_RGB.png"
# payload_path = "test/payload_900B.bin"
# stego_out_path = "stego_rgb_100x100.png"

# k = 2
# top_percent = 30
# key = "secret"

# with open(cover_path, "rb") as f:
#     cover_bytes = f.read()
# with open(payload_path, "rb") as f:
#     payload_bytes = f.read()

# # build flat cover + stable indices
# flat_cover, shape, mode = image_to_flat(cover_bytes, mode="RGB")
# eligible = select_complex_indices_by_key(cover_bytes, mode="RGB", key=key)


# bits_needed = (16 + len(payload_bytes)) * 8
# bits_available = len(eligible) * k
# if bits_needed > bits_available:
#     raise SystemExit("Payload too large")

# stego_flat = embed_xor_lsb_at_indices(flat_cover, payload_bytes, k=k, key=key, indices=eligible)

# stego_bytes = flat_to_image(stego_flat, shape, mode)
# with open(stego_out_path, "wb") as f:
#     f.write(stego_bytes)
# print(f"Stego saved to {stego_out_path}")

# recovered = extract_xor_lsb_at_indices(stego_flat, k=k, key=key, indices=eligible)
# print("Payload match?", recovered == payload_bytes)

# # Pixel Intensity Plot
# cover_path = "test/img_rgb_100x100_RGB.png"
# stego_out_path = "stego_rgb_100x100.png"

# # Convert to grayscale
# cover_img = Image.open(cover_path).convert("L")  
# stego_img = Image.open(stego_out_path).convert("L")

# # Convert to NumPy arrays
# cover_arr = np.array(cover_img).ravel()
# stego_arr = np.array(stego_img).ravel()

# # # Plot histograms
# plt.figure(figsize=(12, 5))

# # 8-bit grayscale image, each pixel intensity is an integer in the range 0–255.
# plt.subplot(1, 2, 1)
# plt.hist(cover_arr, bins=256, range=(0, 255), color="blue", alpha=0.7)
# plt.title("Histogram of Cover Image")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Number of Pixels")

# # For Steg image
# plt.subplot(1, 2, 2)
# plt.hist(stego_arr, bins=256, range=(0, 255), color="green", alpha=0.7)
# plt.title("Histogram of Stego Image")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Number of Pixels")

# plt.tight_layout()
# plt.show()

# RGB Analysis Plot
# cover_img = Image.open("test/img_rgb_100x100_RGB.png").convert("RGB")
# stego_img = Image.open("stego_rgb_100x100.png").convert("RGB")

# # Convert the images to NumPy arrays of shape 
# # Shape = (height, width, 3)
# cover_arr = np.array(cover_img)
# stego_arr = np.array(stego_img)

# channel_names = ["Red", "Green", "Blue"]
# colors = ["red", "green", "blue"]

# plt.figure(figsize=(15, 6))

# # indexes the channel (0=Red, 1=Green, 2=Blue)
# for i, name in enumerate(["Red", "Green", "Blue"]):
#     ax = plt.subplot(1, 3, i+1)
#     c = cover_arr[..., i].ravel()
#     s = stego_arr[..., i].ravel()

#     # Stego: 
#     ax.hist(s, bins=np.arange(257), range=(0, 256),
#             histtype="stepfilled", alpha=1.0, label="Stego", color="yellow")

#     # Cover: 
#     ax.hist(c, bins=np.arange(257), range=(0, 256),
#             histtype="stepfilled", linewidth=1.0, label="Cover", color="black",)

#     ax.set_title(f"{name} channel")
#     ax.set_xlabel("Pixel intensity"); ax.set_ylabel("No. of pixels")
#     ax.set_xlim(0, 255); 
#     ax.legend()

# plt.tight_layout()
# plt.show()

def make_audio_header(payload: bytes, start_sample: int, use_complex: bool) -> bytes:
    sha8 = hashlib.sha256(payload).digest()[:8]
    flags = 1 if use_complex else 0
    return (AudioHeader.magic +
            _struct.pack("<I", len(payload)) + sha8 +
            _struct.pack("<Q", int(start_sample)) +
            _struct.pack("<I", flags) + _struct.pack("<I", 0))

def parse_audio_header(hdr: bytes) -> AudioHeader:
    if len(hdr) < 32 or hdr[:4] != b"STG2":
        raise ValueError("Invalid STG2 audio header")
    length = _struct.unpack("<I", hdr[4:8])[0]
    sha8 = hdr[8:16]
    start_sample = _struct.unpack("<Q", hdr[16:24])[0]
    flags = _struct.unpack("<I", hdr[24:28])[0]
    return AudioHeader(b"STG2", length, sha8, start_sample, flags, 0)

def select_complex_audio_indices(samples: np.ndarray, top_percent: int = 30, window: int = 1024) -> np.ndarray:
    # complexity = moving-avg(|x[n] − x[n−1]|)
    x = samples.astype(np.int64)
    d = np.abs(np.diff(x, prepend=x[:1]))
    if window > 1:
        kernel = np.ones(window, dtype=np.float64) / window
        score = np.convolve(d.astype(np.float64), kernel, mode="same")
    else:
        score = d.astype(np.float64)
    n = score.shape[0]
    k = max(1, int(n * (top_percent / 100.0)))
    idx_sorted = np.argsort(score)
    top_idx = idx_sorted[-k:]
    return np.sort(top_idx).astype(np.int64)

def build_indices_for_audio_with_start(
    samples: np.ndarray,
    key: str,
    k_bits: int,
    payload_nbytes: int,
    start_sample: int,
    use_complex: bool = False,
    complex_top_percent: int = 30,
) -> tuple[np.ndarray, int]:
    n = samples.size
    seed = np.frombuffer(key.encode("utf-8"), dtype=np.uint8).sum(dtype=np.uint32)
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    perm = np.arange(n, dtype=np.int64); rng.shuffle(perm)

    header_bits = 32 * 8  # STG2 header
    hdr_groups = (header_bits + k_bits - 1) // k_bits
    if hdr_groups > n: raise ValueError("Cover too small for audio header")
    hdr_idx = perm[:hdr_groups]

    if use_complex:
        complex_idx = select_complex_audio_indices(samples, top_percent=complex_top_percent)
        hdr_set = set(int(i) for i in hdr_idx)
        complex_idx = np.array([i for i in complex_idx if int(i) not in hdr_set], dtype=np.int64)
        rng2 = np.random.RandomState((int(seed) ^ 0xA5A5A5) & 0x7FFFFFFF)
        rng2.shuffle(complex_idx)
        payload_idx = complex_idx
    else:
        pos = int(np.where(perm == int(start_sample))[0][0]) if int(start_sample) < n else 0
        rotated = np.concatenate([perm[pos:], perm[:pos]])
        hdr_set = set(int(i) for i in hdr_idx)
        payload_idx = np.array([i for i in rotated if int(i) not in hdr_set], dtype=np.int64)

    payload_bits = payload_nbytes * 8
    payload_groups = (payload_bits + k_bits - 1) // k_bits

    if hdr_groups + payload_groups > hdr_idx.size + payload_idx.size:
        raise ValueError("Not enough capacity in selected indices")

    indices_all = np.concatenate([hdr_idx, payload_idx[:payload_groups]]).astype(np.int64)
    return indices_all, hdr_groups

def embed_xor_lsb_audio(samples: np.ndarray, payload: bytes, k: int, key: str,
                           start_sample: int, use_complex: bool = False) -> np.ndarray:
    from lsb_xor_algorithm import keystream_bits  # reuse existing keystream
    indices_all, hdr_groups = build_indices_for_audio_with_start(
        samples, key, k, len(payload), start_sample, use_complex
    )
    hdr_bytes = make_audio_header(payload, start_sample, use_complex)
    hdr_bits = np.unpackbits(np.frombuffer(hdr_bytes, dtype=np.uint8))
    pay_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    C_hdr = hdr_bits ^ keystream_bits(key, hdr_bits.size)
    C_pay = pay_bits ^ keystream_bits(key, pay_bits.size)

    stego = samples.copy()
    # write header
    for i in range((hdr_bits.size + k - 1) // k):
        idx = int(indices_all[i]); chunk = 0
        for t in range(k):
            b = i * k + t
            chunk = (chunk << 1) | (int(C_hdr[b]) if b < hdr_bits.size else 0)
        stego[idx] = (int(stego[idx]) & ~((1 << k) - 1)) | chunk
    # write payload
    off = (hdr_bits.size + k - 1) // k
    need = (pay_bits.size + k - 1) // k

    if off + need > indices_all.size:
        raise ValueError(
            f"Index plan/loop mismatch: need {off + need} slots "
            f"(header={off}, payload={need}), but only "
            f"{indices_all.size} planned"
        )

    for i in range(need):
        idx = int(indices_all[off + i]); chunk = 0
        for t in range(k):
            b = i * k + t
            chunk = (chunk << 1) | (int(C_pay[b]) if b < pay_bits.size else 0)
        stego[idx] = (int(stego[idx]) & ~((1 << k) - 1)) | chunk
    return stego

def extract_xor_lsb_audio(stego: np.ndarray, k: int, key: str):
    from lsb_xor_algorithm import keystream_bits
    n = stego.size
    seed = np.frombuffer(key.encode("utf-8"), dtype=np.uint8).sum(dtype=np.uint32)
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    perm = np.arange(n, dtype=np.int64); rng.shuffle(perm)

    hdr_bits_len = 32 * 8
    hdr_groups = (hdr_bits_len + k - 1) // k
    if hdr_groups > n: raise ValueError("Cover too small for STG2 header")
    hdr_idx = perm[:hdr_groups]

    # read header
    bits = []
    for i in range(hdr_groups):
        v = int(stego[int(hdr_idx[i])]) & ((1 << k) - 1)
        for t in range(k - 1, -1, -1): bits.append((v >> t) & 1)
    C_hdr = np.array(bits[:hdr_bits_len], dtype=np.uint8)
    M_hdr = C_hdr ^ keystream_bits(key, len(C_hdr))
    hdr = parse_audio_header(np.packbits(M_hdr).tobytes())

    # rebuild indices incl. payload
    indices_all, hdr_groups_check = build_indices_for_audio_with_start(
        stego, key, k, hdr.length, start_sample=int(hdr.start_sample),
        use_complex=bool(hdr.flags & 1)
    )
    assert hdr_groups_check == hdr_groups

    # read payload
    pay_groups = (hdr.length * 8 + k - 1) // k
    off = hdr_groups; bits = []
    for i in range(pay_groups):
        v = int(stego[int(indices_all[off + i])]) & ((1 << k) - 1)
        for t in range(k - 1, -1, -1): bits.append((v >> t) & 1)
    C_pay = np.array(bits[: hdr.length * 8], dtype=np.uint8)
    payload = np.packbits(C_pay ^ keystream_bits(key, len(C_pay))).tobytes()
    if hashlib.sha256(payload).digest()[:8] != hdr.sha8:
        raise ValueError("Checksum mismatch")
    return payload, hdr

# =====================================
# ======= EMBED / EXTRACT (MP4) =====
# =====================================

_STG_UUID = uuid.UUID("8d3f5b62-6c6e-4b2c-8e9c-9cc3a9a5b1d2").bytes  # pick your own GUID

def _make_box(box_type: bytes, payload: bytes) -> bytes:
    size = 8 + len(payload)
    return _struct.pack(">I4s", size, box_type) + payload

def _iter_top_level_boxes(b: bytes):
    off, n = 0, len(b)
    while off + 8 <= n:
        size = int.from_bytes(b[off:off+4], "big")
        typ  = b[off+4:off+8]
        if size < 8 or off + size > n:  # malformed
            break
        yield off, size, typ
        off += size

def embed_uuid_box_mp4(mp4_bytes: bytes, payload: bytes, k: int, key: str) -> bytes:
    """
    Hide header+payload in a top-level MP4 uuid box (ignored by players).
    We reuse your existing keystream and audio-style header for simplicity.
    """
    import numpy as np
    # Reuse your header/keystream utilities
    try:
        from lsb_xor_algorithm import make_audio_header, keystream_bits, parse_audio_header
    except Exception:
        # Fallback: minimal 32-byte header = b'STG2' + payload_len(8 LE) + reserved pad
        def make_audio_header(data: bytes, start_sample=0, use_complex=False):
            tag = b"STG2"
            ln  = len(data).to_bytes(8, "little")
            pad = bytes(32 - (4 + 8))
            return tag + ln + pad
        def parse_audio_header(hdr: bytes):
            if not hdr.startswith(b"STG2"):
                raise ValueError("Invalid STG2 audio header")
            length = int.from_bytes(hdr[4:12], "little")
            class H: pass
            h = H(); h.length = length
            return h
        def keystream_bits(key: str, n_bits: int):
            import numpy as np, hashlib
            # simple PRG from SHA256 blocks
            out = bytearray()
            counter = 0
            seed = key.encode("utf-8")
            while len(out) < (n_bits + 7) // 8:
                out += hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
                counter += 1
            bits = np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))
            return bits[:n_bits].astype(np.uint8)

    hdr = make_audio_header(payload, start_sample=0, use_complex=False)  # 32 bytes in your code
    bits = np.unpackbits(np.frombuffer(hdr + payload, dtype=np.uint8))
    C    = bits ^ keystream_bits(key, bits.size)
    cipher = np.packbits(C).tobytes()

    box_payload = _STG_UUID + cipher
    box = _make_box(b"uuid", box_payload)
    return mp4_bytes + box  # append as top-level box (safe)

def extract_uuid_box_mp4(mp4_bytes: bytes, k: int = 0, key: str = "") -> bytes:
    """
    Find our uuid box, keystream-decrypt, parse header, return payload bytes.
    """
    import numpy as np
    from lsb_xor_algorithm import keystream_bits, parse_audio_header  # prefer your real ones
    for off, size, typ in _iter_top_level_boxes(mp4_bytes):
        if typ == b"uuid" and mp4_bytes[off+8:off+24] == _STG_UUID:
            cipher = mp4_bytes[off+24: off+size]
            bits   = np.unpackbits(np.frombuffer(cipher, dtype=np.uint8))
            M      = bits ^ keystream_bits(key, bits.size)
            clear  = np.packbits(M).tobytes()
            hdr    = parse_audio_header(clear[:32])
            payload = clear[32:32+hdr.length]
            return payload
    raise ValueError("No stego uuid box found")

def _iter_mp4_top_level_boxes(b: bytes):
    """
    Yield (offset, size, typ, header_size). Handles 32-bit and 64-bit (size == 1) boxes.
    """
    i, n = 0, len(b)
    while i + 8 <= n:
        size = int.from_bytes(b[i:i+4], "big")
        typ  = b[i+4:i+8]
        if size == 0:
            # box extends to eof
            box_size = n - i
            yield i, box_size, typ, 8
            return
        elif size == 1:
            if i + 16 > n: break
            largesize = int.from_bytes(b[i+8:i+16], "big")
            if largesize < 16: break
            yield i, largesize, typ, 16
            i += largesize
        else:
            if size < 8 or i + size > n: break
            yield i, size, typ, 8
            i += size

def _find_mdat_regions(mp4_bytes: bytes):
    """
    Return list of (data_offset, data_size) for each mdat's payload (excludes header).
    """
    out = []
    for off, size, typ, hdr in _iter_mp4_top_level_boxes(mp4_bytes):
        if typ == b"mdat":
            data_off  = off + hdr
            data_size = max(0, size - hdr)
            if data_off + data_size <= len(mp4_bytes) and data_size > 0:
                out.append((data_off, data_size))
    return out

def _build_indices_for_mp4(mdat_len: int, key: str, k_bits: int, payload_nbytes: int):
    """
    Mirror the audio index planner: a keyed permutation over [0, mdat_len),
    reserve first hdr_groups positions for the header, then payload groups after.
    """
    header_bits = 32 * 8  # we reuse the 32-byte STG2 audio header
    hdr_groups  = (header_bits + k_bits - 1) // k_bits
    payload_bits   = payload_nbytes * 8
    payload_groups = (payload_bits + k_bits - 1) // k_bits

    if hdr_groups + payload_groups > mdat_len:
        raise ValueError("Not enough capacity in mdat for header + payload")

    seed = np.frombuffer(key.encode("utf-8"), dtype=np.uint8).sum(dtype=np.uint32)
    rng  = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    perm = np.arange(mdat_len, dtype=np.int64)
    rng.shuffle(perm)

    indices_all = perm[: hdr_groups + payload_groups].astype(np.int64)
    return indices_all, hdr_groups, payload_groups

def _lsb_write_chunks(byte_arr: bytearray, abs_positions: np.ndarray, bits: np.ndarray, k: int):
    """
    Write 'bits' into k-LSBs of bytes at abs_positions (grouped k bits per position).
    """
    pad = (-len(bits)) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    groups = bits.reshape(-1, k)
    if groups.shape[0] > abs_positions.size:
        raise ValueError("Index plan too small for bits being written")

    mask = ~((1 << k) - 1) & 0xFF
    for i, grp in enumerate(groups):
        v = 0
        for b in grp:
            v = (v << 1) | int(b)
        p = int(abs_positions[i])
        byte_arr[p] = (byte_arr[p] & mask) | v

def _lsb_read_chunks(byte_seq: bytes, abs_positions: np.ndarray, k: int, n_bits: int) -> np.ndarray:
    groups = (n_bits + k - 1) // k
    if groups > abs_positions.size:
        raise ValueError("Index plan too small for bits being read")
    out = []
    mask = (1 << k) - 1
    for i in range(groups):
        p = int(abs_positions[i])
        v = byte_seq[p] & mask
        for t in range(k - 1, -1, -1):
            out.append((v >> t) & 1)
    return np.array(out[:n_bits], dtype=np.uint8)

def embed_xor_lsb_mp4(mp4_bytes: bytes, payload: bytes, k: int, key: str) -> bytes:
    """
    LSB-embed payload in the first (or largest) mdat using the same STG2+XOR design as audio.
    """
    mdats = _find_mdat_regions(mp4_bytes)
    if not mdats:
        raise ValueError("No mdat box found in MP4")
    # choose the largest mdat to maximize capacity
    data_off, data_len = max(mdats, key=lambda t: t[1])

    # plan indices within mdat
    indices_rel, hdr_groups, pay_groups = _build_indices_for_mp4(data_len, key, k, len(payload))
    # lift to absolute file offsets
    abs_idx = data_off + indices_rel

    # header + payload bits (XOR keystream)
    from lsb_xor_algorithm import keystream_bits  # you already have this
    hdr_bytes = make_audio_header(payload, start_sample=0, use_complex=False)  # reuse audio header (32B)
    hdr_bits  = np.unpackbits(np.frombuffer(hdr_bytes, dtype=np.uint8))
    pay_bits  = np.unpackbits(np.frombuffer(payload,    dtype=np.uint8))
    C_hdr = hdr_bits ^ keystream_bits(key, hdr_bits.size)
    C_pay = pay_bits ^ keystream_bits(key, pay_bits.size)

    out = bytearray(mp4_bytes)
    _lsb_write_chunks(out, abs_idx[:hdr_groups], C_hdr, k)
    _lsb_write_chunks(out, abs_idx[hdr_groups:hdr_groups+pay_groups], C_pay, k)
    return bytes(out)

def extract_xor_lsb_mp4(mp4_bytes: bytes, k: int, key: str) -> bytes:
    """
    Recover payload from mdat using the same keyed permutation and 32-byte STG2 (audio) header.
    """
    mdats = _find_mdat_regions(mp4_bytes)
    if not mdats:
        raise ValueError("No mdat box found in MP4")
    data_off, data_len = max(mdats, key=lambda t: t[1])

    # first, read the header using the same index plan length for header only
    header_bits = 32 * 8
    hdr_groups  = (header_bits + k - 1) // k
    # we don't yet know payload length, so build a full perm then slice
    seed = np.frombuffer(key.encode("utf-8"), dtype=np.uint8).sum(dtype=np.uint32)
    rng  = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    perm = np.arange(data_len, dtype=np.int64); rng.shuffle(perm)
    abs_hdr_idx = data_off + perm[:hdr_groups]

    from lsb_xor_algorithm import keystream_bits
    C_hdr = _lsb_read_chunks(mp4_bytes, abs_hdr_idx, k, header_bits)
    M_hdr = C_hdr ^ keystream_bits(key, len(C_hdr))
    hdr   = parse_audio_header(np.packbits(M_hdr).tobytes())  # reuse audio STG2 header parser

    # now know length; rebuild full plan and read payload
    indices_rel, hdr_groups2, pay_groups = _build_indices_for_mp4(data_len, key, k, hdr.length)
    assert hdr_groups2 == hdr_groups
    abs_pay_idx = data_off + indices_rel[hdr_groups:hdr_groups+pay_groups]

    C_pay = _lsb_read_chunks(mp4_bytes, abs_pay_idx, k, hdr.length * 8)
    payload = np.packbits(C_pay ^ keystream_bits(key, len(C_pay))).tobytes()

    import hashlib as _hashlib
    if _hashlib.sha256(payload).digest()[:8] != hdr.sha8:
        raise ValueError("Checksum mismatch")
    return payload
