// Reveal obfuscated email addresses (XOR cipher, hex-encoded in data-c).
(() => {
  const k = 0x5a;
  document.querySelectorAll("[data-c]").forEach((el) => {
    const h = el.dataset.c;
    let s = "";
    for (let i = 0; i < h.length; i += 2) {
      s += String.fromCharCode(parseInt(h.slice(i, i + 2), 16) ^ k);
    }
    el.href = "mailto:" + s;
    el.textContent = s;
  });
})();
