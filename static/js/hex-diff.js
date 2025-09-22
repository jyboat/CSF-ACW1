function initHexDiff() {
  const src = document.getElementById("hex-sources");
  if (!src) return;

  const leftURL = src.dataset.left;
  const rightURL = src.dataset.right;
  if (!leftURL || !rightURL) return;

  const hexL = document.getElementById("hexLeft");
  const hexR = document.getElementById("hexRight");
  const metaL = document.getElementById("hexMetaLeft");
  const metaR = document.getElementById("hexMetaRight");
  const onlyDiffs = document.getElementById("hexOnlyDiffs");

  const printable = (n) =>
    n >= 0x20 && n <= 0x7e ? String.fromCharCode(n) : ".";

  const h2 = (n) => n.toString(16).padStart(2, "0").toUpperCase();
  const h8 = (n) => n.toString(16).padStart(8, "0").toUpperCase();

  function fetchBytes(url) {
    return fetch(url).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b));
  }

  // Render one pane (address | bytes | '|' | ascii)
  function render(container, meta, name, bytes) {
    container.innerHTML = "";
    meta.textContent = `${name} — ${bytes.length.toLocaleString()} bytes`;
    const frag = document.createDocumentFragment();
    const perRow = 16;
    const rows = Math.ceil(bytes.length / perRow) || 1;

    for (let row = 0; row < rows; row++) {
      const off = row * perRow;

      const rowEl = document.createElement("div");
      rowEl.className = "row";

      const addr = document.createElement("div");
      addr.className = "addr";
      addr.textContent = h8(off);

      const bytesEl = document.createElement("div");
      bytesEl.className = "bytes";

      const mid = document.createElement("div");
      mid.className = "mid";
      mid.textContent = "|";

      const ascii = document.createElement("div");
      ascii.className = "ascii";

      // Build both HEX and ASCII with one <span.byte> per position
      for (let i = 0; i < perRow; i++) {
        const idx = off + i;
        const b = idx < bytes.length ? bytes[idx] : null;

        // hex cell
        const hexSpan = document.createElement("span");
        hexSpan.className = "byte";
        hexSpan.dataset.index = idx;
        hexSpan.textContent = b === null ? "  " : h2(b);
        bytesEl.appendChild(hexSpan);

        // ascii cell
        const asciiSpan = document.createElement("span");
        asciiSpan.className = "byte";
        asciiSpan.dataset.index = idx;
        asciiSpan.textContent = b === null ? " " : printable(b);
        ascii.appendChild(asciiSpan);
      }

      rowEl.append(addr, bytesEl, mid, ascii);
      frag.appendChild(rowEl);
    }

    container.appendChild(frag);
  }

  function compare(a, b) {
    const perRow = 16;
    const maxLen = Math.max(a.length, b.length);
    const rows = Math.ceil(maxLen / perRow) || 1;

    const L = hexL.querySelectorAll(".bytes .byte");
    const R = hexR.querySelectorAll(".bytes .byte");

    const rowDiffL = new Set();
    const rowDiffR = new Set();

    for (let i = 0; i < rows * perRow; i++) {
      const av = i < a.length ? a[i] : null;
      const bv = i < b.length ? b[i] : null;

      const lHex = L[i];
      const rHex = R[i];

      if (av === null && bv === null) continue;

      const rowIndex = Math.floor(i / perRow);

      if (av === null && bv !== null) {
        if (rHex) rHex.className = "byte ins";
        markAscii(hexR, i, printable(bv), "ins");
        rowDiffR.add(rowIndex);
      } else if (av !== null && bv === null) {
        if (lHex) lHex.className = "byte del";
        markAscii(hexL, i, printable(av), "del");
        rowDiffL.add(rowIndex);
      } else if (av !== bv) {
        if (lHex) lHex.className = "byte diff";
        if (rHex) rHex.className = "byte diff";
        markAscii(hexL, i, printable(av), "diff");
        markAscii(hexR, i, printable(bv), "diff");
        rowDiffL.add(rowIndex);
        rowDiffR.add(rowIndex);
      }
    }

    const rowsL = hexL.querySelectorAll(".row");
    const rowsR = hexR.querySelectorAll(".row");
    for (let r = 0; r < rows; r++) {
      const same = !(rowDiffL.has(r) || rowDiffR.has(r));
      if (same) {
        rowsL[r]?.classList.add("same-row");
        rowsR[r]?.classList.add("same-row");
      }
    }
  }

  function markAscii(container, idx, char, kind) {
    const perRow = 16;
    const row = Math.floor(idx / perRow);
    const col = idx % perRow;

    const rowEl = container.querySelectorAll(".row")[row];
    if (!rowEl) return;

    const asciiSpans = rowEl.querySelectorAll(".ascii .byte");
    const cell = asciiSpans[col];
    if (!cell) return;

    cell.className = `byte ${kind}`;
    cell.textContent = char;
  }

  // Row-based scroll sync so panes stay aligned
  function setupRowSync(leftEl, rightEl) {
    let isSyncing = false;

    function sync(master, other) {
      if (isSyncing) return;
      isSyncing = true;

      const rowH = master.querySelector(".row")?.offsetHeight || 1;
      const rowIdx = Math.round(master.scrollTop / rowH);

      other.scrollTop = rowIdx * rowH;

      isSyncing = false;
    }

    leftEl.addEventListener("scroll", () => sync(leftEl, rightEl));
    rightEl.addEventListener("scroll", () => sync(rightEl, leftEl));
  }

  // Fetch both, render, then diff and wire up UI
  Promise.all([fetchBytes(leftURL), fetchBytes(rightURL)]).then(([A, B]) => {
    render(hexL, metaL, new URL(leftURL, location.href).pathname.split("/").pop(), A);
    render(hexR, metaR, new URL(rightURL, location.href).pathname.split("/").pop(), B);

    compare(A, B);
    setupRowSync(hexL, hexR);

    // “show only rows with diffs”
    if (onlyDiffs) {
      onlyDiffs.addEventListener("change", () => {
        [hexL, hexR].forEach((h) =>
          h.classList.toggle("only-diffs", onlyDiffs.checked)
        );
        // re-align after filter toggles
        requestAnimationFrame(() => {
          hexL.scrollTop = hexL.scrollTop;
        });
      });
    }
  });
}