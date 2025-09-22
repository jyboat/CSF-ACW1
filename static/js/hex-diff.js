let left = null;
let right = null;

const PER_ROW = 16;
const ROW_HEIGHT = 20; // px — adjust if your CSS row height differs
const BUFFER_ROWS = 150; // how many rows to render at once

function initHexDiff() {
  const sources = document.getElementById("hex-sources");
  left = sources.dataset.left;
  right = sources.dataset.right;

  if (!left || !right) {
    console.error("HexDiff: missing file URLs", left, right);
    return;
  }

  const hexL = document.getElementById("hexLeft");
  const hexR = document.getElementById("hexRight");
  const metaL = document.getElementById("hexMetaLeft");
  const metaR = document.getElementById("hexMetaRight");
  const onlyDiffs = document.getElementById("hexOnlyDiffs");
  const progressBar = document.getElementById("hexProgressBar");

  const printable = (n) =>
    n >= 0x20 && n <= 0x7e ? String.fromCharCode(n) : ".";
  const h2 = (n) => n.toString(16).padStart(2, "0").toUpperCase();
  const h8 = (n) => n.toString(16).padStart(8, "0").toUpperCase();

  async function fetchBytesWithProgress(url) {
    const response = await fetch(url);
    const reader = response.body.getReader();
    const contentLength = +response.headers.get("Content-Length");

    let receivedLength = 0;
    let chunks = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      receivedLength += value.length;

      if (contentLength) {
        const percent = Math.round((receivedLength / contentLength) * 100);
        progressBar.style.width = percent + "%";
        progressBar.textContent = percent + "%";
      }
    }

    let merged = new Uint8Array(receivedLength);
    let pos = 0;
    for (let chunk of chunks) {
      merged.set(chunk, pos);
      pos += chunk.length;
    }

    return merged;
  }

  function buildRow(rowIdx, A, B) {
    const off = rowIdx * PER_ROW;

    const rowL = document.createElement("div");
    rowL.className = "row";
    rowL.style.top = `${rowIdx * ROW_HEIGHT}px`;
    rowL.style.position = "absolute";

    const rowR = rowL.cloneNode(false);

    const addrL = document.createElement("div");
    addrL.className = "addr";
    addrL.textContent = h8(off);
    const addrR = addrL.cloneNode(true);

    const bytesL = document.createElement("div");
    bytesL.className = "bytes";
    const midL = document.createElement("div");
    midL.className = "mid";
    midL.textContent = "|";
    const asciiL = document.createElement("div");
    asciiL.className = "ascii";

    const bytesR = document.createElement("div");
    bytesR.className = "bytes";
    const midR = document.createElement("div");
    midR.className = "mid";
    midR.textContent = "|";
    const asciiR = document.createElement("div");
    asciiR.className = "ascii";

    let rowHasDiff = false;

    for (let i = 0; i < PER_ROW; i++) {
      const idx = off + i;

      const av = idx < A.length ? A[idx] : null;
      const bv = idx < B.length ? B[idx] : null;

      const lHex = document.createElement("span");
      lHex.className = "byte";
      lHex.textContent = av === null ? "  " : h2(av);

      const lAsc = document.createElement("span");
      lAsc.className = "byte";
      lAsc.textContent = av === null ? " " : printable(av);

      const rHex = document.createElement("span");
      rHex.className = "byte";
      rHex.textContent = bv === null ? "  " : h2(bv);

      const rAsc = document.createElement("span");
      rAsc.className = "byte";
      rAsc.textContent = bv === null ? " " : printable(bv);

      if (av === null && bv !== null) {
        rHex.className = "byte ins"; rAsc.className = "byte ins";
        rowHasDiff = true;
      } else if (av !== null && bv === null) {
        lHex.className = "byte del"; lAsc.className = "byte del";
        rowHasDiff = true;
      } else if (av !== bv) {
        lHex.className = "byte diff"; rHex.className = "byte diff";
        lAsc.className = "byte diff"; rAsc.className = "byte diff";
        rowHasDiff = true;
      }

      bytesL.appendChild(lHex); asciiL.appendChild(lAsc);
      bytesR.appendChild(rHex); asciiR.appendChild(rAsc);
    }

    if (!rowHasDiff) {
      rowL.classList.add("same-row");
      rowR.classList.add("same-row");
    }

    rowL.append(addrL, bytesL, midL, asciiL);
    rowR.append(addrR, bytesR, midR, asciiR);

    return [rowL, rowR];
  }

  function renderVirtual(A, B) {
    const totalRows = Math.ceil(Math.max(A.length, B.length) / PER_ROW) || 1;

    // precompute which rows have diffs
    const diffRows = new Set();
    for (let row = 0; row < totalRows; row++) {
      const off = row * PER_ROW;
      let hasDiff = false;
      for (let i = 0; i < PER_ROW; i++) {
        const idx = off + i;
        const av = idx < A.length ? A[idx] : null;
        const bv = idx < B.length ? B[idx] : null;
        if (av !== bv) { hasDiff = true; break; }
      }
      if (hasDiff) diffRows.add(row);
    }

    // setup spacers
    hexL.style.position = "relative";
    hexR.style.position = "relative";
    hexL.innerHTML = "";
    hexR.innerHTML = "";

    const spacerL = document.createElement("div");
    const spacerR = document.createElement("div");

    function updateSpacerHeight() {
      const rowsToShow = onlyDiffs.checked ? diffRows.size : totalRows;
      spacerL.style.height = `${rowsToShow * ROW_HEIGHT}px`;
      spacerR.style.height = `${rowsToShow * ROW_HEIGHT}px`;
    }
    updateSpacerHeight();

    hexL.appendChild(spacerL);
    hexR.appendChild(spacerR);

    function renderViewport() {
      const scrollTop = hexL.scrollTop;
      const startRow = Math.floor(scrollTop / ROW_HEIGHT);
      const endRow = startRow + BUFFER_ROWS;

      hexL.innerHTML = "";
      hexR.innerHTML = "";
      hexL.appendChild(spacerL);
      hexR.appendChild(spacerR);

      const fragL = document.createDocumentFragment();
      const fragR = document.createDocumentFragment();

      let rows = onlyDiffs.checked ? Array.from(diffRows) : [...Array(totalRows).keys()];
      rows = rows.slice(startRow, endRow);

      rows.forEach((row, i) => {
        const [rowL, rowR] = buildRow(row, A, B);
        const top = (startRow + i) * ROW_HEIGHT;
        rowL.style.top = `${top}px`;
        rowR.style.top = `${top}px`;
        fragL.appendChild(rowL);
        fragR.appendChild(rowR);
      });

      hexL.appendChild(fragL);
      hexR.appendChild(fragR);
    }

    hexL.addEventListener("scroll", () => {
      hexR.scrollTop = hexL.scrollTop;
      renderViewport();
    });
    hexR.addEventListener("scroll", () => {
      hexL.scrollTop = hexR.scrollTop;
      renderViewport();
    });

    if (onlyDiffs) {
      onlyDiffs.addEventListener("change", () => {
        updateSpacerHeight();
        hexL.scrollTop = 0;
        hexR.scrollTop = 0;
        renderViewport();
      });
    }

    renderViewport();
  }

  Promise.all([fetchBytesWithProgress(left), fetchBytesWithProgress(right)]).then(([A, B]) => {
    metaL.textContent = `Cover — ${A.length.toLocaleString()} bytes`;
    metaR.textContent = `Stego — ${B.length.toLocaleString()} bytes`;

    renderVirtual(A, B);

    document.getElementById("hexLoading").style.display = "none";

    if (onlyDiffs) {
      onlyDiffs.addEventListener("change", () => {
        [hexL, hexR].forEach((pane) =>
          pane.classList.toggle("only-diffs", onlyDiffs.checked)
        );
        renderVirtual(A, B);
      });
    }
  });
}
