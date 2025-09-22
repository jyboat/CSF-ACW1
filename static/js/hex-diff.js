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

  function readBuf(url) {
    return fetch(url).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b));
  }

  function render(container, meta, name, bytes) {
    container.innerHTML = "";
    meta.textContent = `${name} — ${bytes.length.toLocaleString()} bytes`;
    const frag = document.createDocumentFragment();
    const perRow = 16,
      rows = Math.ceil(bytes.length / perRow) || 1;

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

      let asciiLine = "";
      for (let i = 0; i < perRow; i++) {
        const idx = off + i;
        const b = idx < bytes.length ? bytes[idx] : null;
        const span = document.createElement("span");
        span.className = "byte";
        span.dataset.index = idx;
        span.textContent = b === null ? "  " : h2(b);
        bytesEl.appendChild(span);
        asciiLine += b === null ? " " : printable(b);
      }
      ascii.textContent = asciiLine;

      rowEl.append(addr, bytesEl, mid, ascii);
      frag.appendChild(rowEl);
    }
    container.appendChild(frag);
  }

  function compare(a, b) {
    const perRow = 16,
      maxLen = Math.max(a.length, b.length);
    const rows = Math.ceil(maxLen / perRow) || 1;
    const L = hexL.querySelectorAll(".byte");
    const R = hexR.querySelectorAll(".byte");

    const rowDiffL = new Set(),
      rowDiffR = new Set();

    for (let i = 0; i < rows * perRow; i++) {
      const av = i < a.length ? a[i] : null;
      const bv = i < b.length ? b[i] : null;
      const l = L[i],
        r = R[i];
      if (!l && !r) continue;

      if (av === null && bv !== null) {
        if (r) r.className = "byte ins";
        markAscii(hexR, i, printable(bv), "ins");
        rowDiffR.add(Math.floor(i / perRow));
      } else if (bv === null && av !== null) {
        if (l) l.className = "byte del";
        markAscii(hexL, i, printable(av), "del");
        rowDiffL.add(Math.floor(i / perRow));
      } else if (av !== bv) {
        if (l) l.className = "byte diff";
        if (r) r.className = "byte diff";
        markAscii(hexL, i, printable(av), "diff");
        markAscii(hexR, i, printable(bv), "diff");
        rowDiffL.add(Math.floor(i / perRow));
        rowDiffR.add(Math.floor(i / perRow));
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
    const row = Math.floor(idx / 16),
      col = idx % 16;
    const rowEl = container.querySelectorAll(".row")[row];
    if (!rowEl) return;
    const ascii = rowEl.querySelector(".ascii");
    const text = ascii.textContent;
    ascii.textContent = "";
    ascii.append(document.createTextNode(text.slice(0, col)));
    const span = document.createElement("span");
    span.className = `byte ${kind}`;
    span.textContent = char;
    ascii.append(span, document.createTextNode(text.slice(col + 1)));
  }

  // Row-based scroll sync: keep both panes aligned to the same row index
  function setupRowSync(leftEl, rightEl) {
    let isSyncing = false;

    function sync(master, other) {
      if (isSyncing) return;
      isSyncing = true;
      requestAnimationFrame(() => {
        const rowHeight = master.querySelector(".row")?.offsetHeight || 1;
        const rowIndex = Math.floor(master.scrollTop / rowHeight);
        other.scrollTop = rowIndex * rowHeight;
        isSyncing = false;
      });
    }

    leftEl.addEventListener("scroll", () => sync(leftEl, rightEl), { passive: true });
    rightEl.addEventListener("scroll", () => sync(rightEl, leftEl), { passive: true });
  }

  // Load both files, render, compare, then setup scroll sync
  Promise.all([readBuf(leftURL), readBuf(rightURL)]).then(([a, b]) => {
    render(hexL, metaL, leftURL.split("/").pop(), a);
    render(hexR, metaR, rightURL.split("/").pop(), b);
    compare(a, b);

    setupRowSync(hexL, hexR);

    if (onlyDiffs) {
      onlyDiffs.addEventListener("change", () => {
        [hexL, hexR].forEach((h) =>
          h.classList.toggle("only-diffs", onlyDiffs.checked)
        );
        // re-align both after the filter changes
        requestAnimationFrame(() => {
          hexL.scrollTop = hexL.scrollTop; // trigger sync once
        });
      });
    }
  });
}
