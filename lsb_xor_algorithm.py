import io, struct, hashlib, random
from dataclasses import dataclass
import numpy as np
from PIL import Image

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
        block = hashlib.sha256(key.encode() + struct.pack(">Q", counter)).digest()
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
    return h.magic + struct.pack("<I", h.length) + h.sha8

def parse_header(hdr: bytes) -> Header:
    if len(hdr) < 16 or hdr[:4] != b"STG1":
        raise ValueError("Invalid header")
    length = struct.unpack("<I", hdr[4:8])[0]
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
    length = struct.unpack("<I", all_bytes[4:8])[0]
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
