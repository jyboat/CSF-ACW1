import io, struct as _struct, hashlib, random
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

# Image helpers

def image_to_flat(image_bytes: bytes, mode: str = "RGB"):
    im = Image.open(io.BytesIO(image_bytes)).convert(mode)
    arr = np.array(im, dtype=np.uint8)        # (H, W, C) or (H, W)
    return arr.reshape(-1), arr.shape, im.mode

def flat_to_image(flat: np.ndarray, shape: tuple, mode: str = "RGB") -> bytes:
    arr = flat.reshape(shape).astype(np.uint8)
    im = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    im.save(buf, format="PNG")                # lossless
    buf.seek(0)
    return buf.getvalue()

# Edge/texture (complex) selection

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

def select_complex_indices_from_image(image_bytes: bytes, top_percent=30,
                                      mode="RGB", key=None):
    im = Image.open(io.BytesIO(image_bytes)).convert(mode)
    arr = np.array(im, dtype=np.uint8)
    if arr.ndim == 2:
        H, W = arr.shape; C = 1
        gray = arr
    else:
        H, W, C = arr.shape
        if C >= 3:
            gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.uint8)
        else:
            gray = arr.squeeze().astype(np.uint8)

    mag = _sobel_magnitude(gray)
    thresh = np.percentile(mag, 100 - top_percent)
    mask_pix = (mag >= thresh).ravel()
    pix_idx = np.flatnonzero(mask_pix)
    chan_offsets = np.arange(C, dtype=np.int64)
    eligible = (pix_idx[:, None] * C + chan_offsets[None, :]).reshape(-1)

    if key is not None:
        seed = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], 'little')
        rng = random.Random(seed)
        el = eligible.tolist()
        rng.shuffle(el)
        eligible = np.array(el, dtype=np.int64)

    flat = arr.reshape(-1)
    return flat, arr.shape, im.mode, eligible

# XOR keystream + LSB embed/extract (indices-aware)

def keystream_bits(key: str, n_bits: int) -> np.ndarray:
    if n_bits <= 0:
        return np.empty(0, dtype=np.uint8)
    counter, out = 0, []
    while len(out) < n_bits:
        block = hashlib.sha256(key.encode() + _struct.pack(">Q", counter)).digest()
        for byte in block:
            for i in range(8):
                out.append((byte >> (7 - i)) & 1)
                if len(out) >= n_bits:
                    return np.array(out, dtype=np.uint8)
        counter += 1

@dataclass
class Header:
    magic: bytes = b"STG1"
    length: int = 0
    sha8: bytes = b""

def make_header(payload: bytes) -> bytes:
    h = Header(length=len(payload), sha8=hashlib.sha256(payload).digest()[:8])
    return h.magic + _struct.pack("<I", h.length) + h.sha8

def parse_header(hdr: bytes) -> Header:
    if len(hdr) < 16 or hdr[:4] != b"STG1":
        raise ValueError("Invalid header")
    length = _struct.unpack("<I", hdr[4:8])[0]
    sha8 = hdr[8:16]
    return Header(magic=b"STG1", length=length, sha8=sha8)

def embed_xor_lsb_at_indices(cover: np.ndarray, payload: bytes, k: int, key: str,
                             indices: np.ndarray) -> np.ndarray:
    header = make_header(payload)
    onwire = header + payload
    M = np.unpackbits(np.frombuffer(onwire, dtype=np.uint8))
    K = keystream_bits(key, len(M))
    C = M ^ K
    pad = (-len(C)) % k
    if pad:
        C = np.concatenate([C, np.zeros(pad, dtype=np.uint8)])
    groups = C.reshape(-1, k)
    needed = groups.shape[0]
    if needed > len(indices):
        raise ValueError("Not enough complex locations")
    out = cover.copy()
    mask = ~((1 << k) - 1)
    for i, grp in enumerate(groups):
        v = 0
        for b in grp:
            v = (v << 1) | int(b)
        j = int(indices[i])
        out[j] = (int(out[j]) & mask) | v
    return out

def extract_xor_lsb_at_indices(stego: np.ndarray, k: int, key: str, indices: np.ndarray) -> bytes:
    header_bits = 16 * 8
    groups_header = (header_bits + k - 1) // k
    bits = []
    for i in range(groups_header):
        v = int(stego[int(indices[i])]) & ((1 << k) - 1)
        for t in range(k - 1, -1, -1):
            bits.append((v >> t) & 1)
    C_hdr = np.array(bits[:header_bits], dtype=np.uint8)
    K_hdr = keystream_bits(key, len(C_hdr))
    M_hdr = C_hdr ^ K_hdr
    header_bytes = np.packbits(M_hdr).tobytes()
    hdr = parse_header(header_bytes)
    total_bytes = 16 + hdr.length
    total_bits = total_bytes * 8
    groups_total = (total_bits + k - 1) // k
    bits = []
    for i in range(groups_total):
        v = int(stego[int(indices[i])]) & ((1 << k) - 1)
        for t in range(k - 1, -1, -1):
            bits.append((v >> t) & 1)
    C_all = np.array(bits[:total_bits], dtype=np.uint8)
    K_all = keystream_bits(key, len(C_all))
    M_all = C_all ^ K_all
    all_bytes = np.packbits(M_all).tobytes()
    length = _struct.unpack("<I", all_bytes[4:8])[0]
    sha8 = all_bytes[8:16]
    payload = all_bytes[16:16 + length]
    if hashlib.sha256(payload).digest()[:8] != sha8:
        raise ValueError("Checksum mismatch")
    return payload

# Run immediately

cover_path = "test/img_rgb_100x100_RGB.png"
payload_path = "test/payload_900B.bin"
stego_out_path = "stego_rgb_100x100.png"

k = 2
top_percent = 30
key = "secret"

with open(cover_path, "rb") as f:
    cover_bytes = f.read()
with open(payload_path, "rb") as f:
    payload_bytes = f.read()

flat_cover, shape, mode, eligible = select_complex_indices_from_image(
    cover_bytes, top_percent=top_percent, mode="RGB", key=key
)

bits_needed = (16 + len(payload_bytes)) * 8
bits_available = len(eligible) * k
if bits_needed > bits_available:
    raise SystemExit("Payload too large")

stego_flat = embed_xor_lsb_at_indices(flat_cover, payload_bytes, k=k, key=key, indices=eligible)

stego_bytes = flat_to_image(stego_flat, shape, mode)
with open(stego_out_path, "wb") as f:
    f.write(stego_bytes)
print(f"Stego saved to {stego_out_path}")

recovered = extract_xor_lsb_at_indices(stego_flat, k=k, key=key, indices=eligible)
print("Payload match?", recovered == payload_bytes)

# Pixel Intensity Plot
# cover_path = "test/img_rgb_100x100_RGB.png"
# stego_out_path = "stego_rgb_100x100.png"

## Convert to grayscale
# cover_img = Image.open(cover_path).convert("L")  
# stego_img = Image.open(stego_out_path).convert("L")

## Convert to NumPy arrays
# cover_arr = np.array(cover_img).ravel()
# stego_arr = np.array(stego_img).ravel()

# # Plot histograms
# plt.figure(figsize=(12, 5))

## 8-bit grayscale image, each pixel intensity is an integer in the range 0–255.
# plt.subplot(1, 2, 1)
# plt.hist(cover_arr, bins=256, range=(0, 255), color="blue", alpha=0.7)
# plt.title("Histogram of Cover Image")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Number of Pixels")

## For Steg image
# plt.subplot(1, 2, 2)
# plt.hist(stego_arr, bins=256, range=(0, 255), color="green", alpha=0.7)
# plt.title("Histogram of Stego Image")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Number of Pixels")

# plt.tight_layout()
# plt.show()

# RGB Analysis Plot
cover_img = Image.open("test/img_rgb_100x100_RGB.png").convert("RGB")
stego_img = Image.open("stego_rgb_100x100.png").convert("RGB")

# Convert the images to NumPy arrays of shape 
# Shape = (height, width, 3)
cover_arr = np.array(cover_img)
stego_arr = np.array(stego_img)

channel_names = ["Red", "Green", "Blue"]
colors = ["red", "green", "blue"]

plt.figure(figsize=(15, 6))

# indexes the channel (0=Red, 1=Green, 2=Blue)
for i, name in enumerate(["Red", "Green", "Blue"]):
    ax = plt.subplot(1, 3, i+1)
    c = cover_arr[..., i].ravel()
    s = stego_arr[..., i].ravel()

    # Stego: 
    ax.hist(s, bins=np.arange(257), range=(0, 256),
            histtype="stepfilled", alpha=1.0, label="Stego", color="yellow")

    # Cover: 
    ax.hist(c, bins=np.arange(257), range=(0, 256),
            histtype="stepfilled", linewidth=1.0, label="Cover", color="black",)

    ax.set_title(f"{name} channel")
    ax.set_xlabel("Pixel intensity"); ax.set_ylabel("No. of pixels")
    ax.set_xlim(0, 255); 
    ax.legend()

plt.tight_layout()
plt.show()

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

    total_bits = (32 + payload_nbytes) * 8
    total_groups = (total_bits + k_bits - 1) // k_bits
    if total_groups > (hdr_idx.size + payload_idx.size):
        raise ValueError("Not enough capacity in selected indices")

    payload_groups = total_groups - hdr_idx.size
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
    for i in range((pay_bits.size + k - 1) // k):
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