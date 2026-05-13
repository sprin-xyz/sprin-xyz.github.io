(() => {
  // If anything below throws, drop the .js class so `[data-animate]` content
  // becomes visible (otherwise the page would appear blank without JS).
  try {

  // Reveal obfuscated email addresses (XOR cipher, hex-encoded in data-c).
  (function () {
    const k = 0x5a;
    document.querySelectorAll("[data-c]").forEach((el) => {
      const h = el.dataset.c;
      let s = "";
      for (let i = 0; i < h.length; i += 2) {
        s += String.fromCharCode(parseInt(h.substr(i, 2), 16) ^ k);
      }
      el.href = "mailto:" + s;
      el.textContent = s;
    });
  })();

  const path = document.getElementById("trajectory-path");
  const dot = document.getElementById("moving-dot");
  const shadow = document.getElementById("moving-dot-shadow");
  const stage = document.querySelector(".hero-art-wrap");
  const animated = document.querySelectorAll("[data-animate]");

  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const prefersReducedMotion = window.matchMedia
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Path is parameterized so length 0 is the upper-right endpoint and 1 is
  // the lower-left endpoint.
  const TRAVEL_START = 0.06;   // start a bit in from the right endpoint
  const TRAVEL_END = 0.88;     // and travel further than before
  const SLOPE_FLOOR = 0.18;    // baseline speed on the flat parts of the curve
  let length = 0;
  // Precomputed table mapping linear scroll-progress -> path-length offset,
  // weighted so the dot moves faster where the curve is steeper.
  let scrub = null;

  function setupPath() {
    if (!path) return;
    const pathLen = path.getTotalLength();
    // The SVG can be in a layout state (e.g. before the image lays out) where
    // getTotalLength returns 0 — bail rather than building a NaN scrub table.
    if (!pathLen || !isFinite(pathLen)) {
      length = 0;
      scrub = null;
      return;
    }
    length = pathLen;

    const N = 240;
    const startL = length * TRAVEL_START;
    const endL = length * TRAVEL_END;
    const span = endL - startL;
    const eps = Math.max(1, length * 0.0015);

    // Sample weights at each step: w ∝ |dy/dx|, with a floor so flat
    // regions still inch forward instead of stalling.
    const w = new Float64Array(N + 1);
    for (let i = 0; i <= N; i++) {
      const s = startL + (span * i) / N;
      const a = path.getPointAtLength(Math.max(0, s - eps));
      const b = path.getPointAtLength(Math.min(length, s + eps));
      const dx = Math.abs(b.x - a.x);
      const dy = Math.abs(b.y - a.y);
      const slope = dx > 1e-6 ? dy / dx : 10;
      w[i] = SLOPE_FLOOR + slope;
    }
    // Cumulative integral via trapezoid rule.
    const cum = new Float64Array(N + 1);
    for (let i = 1; i <= N; i++) {
      cum[i] = cum[i - 1] + (w[i - 1] + w[i]) / 2;
    }
    const total = cum[N];
    scrub = { N, startL, span, cum, total };
  }

  function offsetFor(p) {
    if (!scrub) return 0;
    const target = scrub.total * p;
    // Binary-search for the first i where cum[i] >= target.
    let lo = 0;
    let hi = scrub.N;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (scrub.cum[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    if (lo === 0) return scrub.startL;
    const c0 = scrub.cum[lo - 1];
    const c1 = scrub.cum[lo];
    const t = c1 > c0 ? (target - c0) / (c1 - c0) : 0;
    return scrub.startL + (scrub.span * (lo - 1 + t)) / scrub.N;
  }

  function setDot(p, opacity) {
    if (!path || !dot || !shadow || !length) return;
    const pp = clamp(p, 0, 1);
    const point = path.getPointAtLength(offsetFor(pp));
    dot.setAttribute("cx", point.x);
    dot.setAttribute("cy", point.y);
    shadow.setAttribute("cx", point.x);
    shadow.setAttribute("cy", point.y);
    dot.style.opacity = String(opacity);
    shadow.style.opacity = String(opacity);
  }

  if (animated.length) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add("is-visible");
      });
    }, { threshold: 0.14 });
    animated.forEach((el) => observer.observe(el));
  }

  // Pure momentum: scroll events impart impulses, friction decays velocity,
  // no target to spring toward. Once the dot drifts past either boundary it
  // fades and stays gone (until reload).
  let position = 0;           // warped progress in [0, 1]-ish (can drift out)
  let velocity = 0;           // warped progress per frame
  let lastScrollY = 0;
  let rafId = null;
  let gone = false;
  const SENSITIVITY = 1 / 3500; // scroll px -> warped progress impulse
  const DAMPING = 0.78;         // velocity retention per frame
  const FADE_RANGE = 0.07;     // overshoot needed for full fade

  function opacityFor(p) {
    if (p < 0) return clamp(1 + p / FADE_RANGE, 0, 1);
    if (p > 1) return clamp(1 - (p - 1) / FADE_RANGE, 0, 1);
    return 1;
  }

  function tick() {
    rafId = null;
    if (gone) return;
    velocity *= DAMPING;
    position += velocity;
    const op = opacityFor(position);
    if (op <= 0) {
      gone = true;
      setDot(clamp(position, 0, 1), 0);
      return;
    }
    setDot(position, op);
    if (Math.abs(velocity) > 1e-5) {
      rafId = requestAnimationFrame(tick);
    }
  }

  function onScroll() {
    if (gone) return;
    const y = window.scrollY || window.pageYOffset || 0;
    const vh = window.innerHeight || document.documentElement.clientHeight;
    // Slow near the top of the page, ramping up smoothly further down.
    const factor = 0.35 + clamp(y / (1.6 * vh), 0, 1) * 0.95;
    velocity += (y - lastScrollY) * SENSITIVITY * factor;
    lastScrollY = y;
    if (rafId === null) rafId = requestAnimationFrame(tick);
  }

  setupPath();
  lastScrollY = window.scrollY || window.pageYOffset || 0;
  setDot(position, 1);

  window.addEventListener("resize", () => {
    setupPath();
    if (!gone) setDot(clamp(position, 0, 1), opacityFor(position));
  });

  if (prefersReducedMotion) {
    // Honor the user's preference: do not animate the dot based on scroll.
    // Render it once at the starting position and leave it there.
  } else {
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  // Browser back/forward cache: when the page is restored from bfcache the
  // scroll position may have been restored too. Re-sync lastScrollY so the
  // first scroll event after restore doesn't compute a huge delta and yeet
  // the dot off-screen.
  window.addEventListener("pageshow", () => {
    lastScrollY = window.scrollY || window.pageYOffset || 0;
    velocity = 0;
  });

  // --- Portrait expand-on-click ---------------------------------------------
  // Alignment computed via ORB feature matching between jacob_poster_crop.jpg
  // and jacob_poster_photo.jpg (similarity transform, 610/700 RANSAC inliers).
  // The crop occupies this sub-rect of the photo (photo is 2048x1920):
  const PHOTO_W = 2048;
  const PHOTO_H = 1920;
  const CROP_X0 = 279.1;
  const CROP_Y0 = 136.5;
  const CROP_W = 1305.2;
  const CROP_H = 1337.1;

  const portrait = document.querySelector(".portrait[data-expandable]");
  let openState = null;

  function startStateFor(rect) {
    // Pick a uniform scale that fits the crop region's *width* into the
    // container, matching the cover-with-AR<1 behavior of the original img.
    const scale = rect.width / CROP_W;
    const photoDispW = PHOTO_W * scale;
    const photoDispH = PHOTO_H * scale;
    const cropCenterX = CROP_X0 + CROP_W / 2;
    const cropCenterY = CROP_Y0 + CROP_H / 2;
    // Place the crop's center at the container's center, then shift up slightly
    // to mirror the original `object-position: 50% 43%`.
    const offsetX = rect.width / 2 - cropCenterX * scale;
    const offsetY = rect.height / 2 - cropCenterY * scale - rect.height * 0.07;
    return {
      bgSizeX: (photoDispW / rect.width) * 100,
      bgSizeY: (photoDispH / rect.height) * 100,
      bgPosX: (-offsetX / (photoDispW - rect.width)) * 100,
      bgPosY: (-offsetY / (photoDispH - rect.height)) * 100,
    };
  }

  function openPortrait() {
    if (!portrait || openState) return;
    const rect = portrait.getBoundingClientRect();
    const s = startStateFor(rect);

    const photoAR = PHOTO_W / PHOTO_H;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const margin = Math.min(vw, vh) * 0.08;
    let targetW = vw - margin * 2;
    let targetH = targetW / photoAR;
    if (targetH > vh - margin * 2) {
      targetH = vh - margin * 2;
      targetW = targetH * photoAR;
    }
    const targetX = (vw - targetW) / 2;
    const targetY = (vh - targetH) / 2;

    const backdrop = document.createElement("div");
    backdrop.className = "photo-backdrop";
    backdrop.setAttribute("role", "dialog");
    backdrop.setAttribute("aria-modal", "true");
    backdrop.setAttribute("aria-label", "Full photo of Jacob Springer");

    const expanded = document.createElement("div");
    expanded.className = "photo-expanded";
    expanded.setAttribute("tabindex", "-1");
    Object.assign(expanded.style, {
      left: rect.left + "px",
      top: rect.top + "px",
      width: rect.width + "px",
      height: rect.height + "px",
      backgroundImage: "url('assets/jacob_poster_photo.jpg')",
      backgroundSize: `${s.bgSizeX}% ${s.bgSizeY}%`,
      backgroundPosition: `${s.bgPosX}% ${s.bgPosY}%`,
      borderRadius: "50%",
    });

    backdrop.appendChild(expanded);
    document.body.appendChild(backdrop);

    // Lock background scrolling while the modal is open.
    const prevBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    // Force a layout flush so the initial styles are committed before we
    // change them — otherwise the transition won't fire.
    void expanded.getBoundingClientRect();

    portrait.classList.add("is-hidden");
    backdrop.classList.add("is-open");
    Object.assign(expanded.style, {
      left: targetX + "px",
      top: targetY + "px",
      width: targetW + "px",
      height: targetH + "px",
      backgroundSize: "100% 100%",
      backgroundPosition: "0% 0%",
      borderRadius: "12px",
    });

    expanded.focus({ preventScroll: true });

    function onBackdropClick(e) {
      if (e.target === expanded) return; // click on photo: do nothing
      closePortrait();
    }
    function onKey(e) {
      if (e.key === "Escape") {
        closePortrait();
      } else if (e.key === "Tab") {
        // Only one focusable element inside; keep focus pinned to it.
        e.preventDefault();
        expanded.focus({ preventScroll: true });
      }
    }
    backdrop.addEventListener("click", onBackdropClick);
    document.addEventListener("keydown", onKey);

    openState = { backdrop, expanded, onKey, prevBodyOverflow };
  }

  function closePortrait() {
    if (!openState) return;
    const { backdrop, expanded, onKey, prevBodyOverflow } = openState;
    document.removeEventListener("keydown", onKey);
    document.body.style.overflow = prevBodyOverflow;

    const rect = portrait.getBoundingClientRect();
    const s = startStateFor(rect);

    backdrop.classList.remove("is-open");
    Object.assign(expanded.style, {
      left: rect.left + "px",
      top: rect.top + "px",
      width: rect.width + "px",
      height: rect.height + "px",
      backgroundSize: `${s.bgSizeX}% ${s.bgSizeY}%`,
      backgroundPosition: `${s.bgPosX}% ${s.bgPosY}%`,
      borderRadius: "50%",
    });

    const cleanup = () => {
      backdrop.remove();
      portrait.classList.remove("is-hidden");
      openState = null;
      // Return focus to the trigger.
      portrait.focus({ preventScroll: true });
    };
    expanded.addEventListener("transitionend", cleanup, { once: true });
    // Fallback in case transitionend doesn't fire (e.g. prefers-reduced-motion):
    setTimeout(() => {
      if (openState && openState.backdrop === backdrop) cleanup();
    }, 700);
  }

  function onPortraitActivate(e) {
    // Click already triggers openPortrait via the click handler; this
    // handler covers Enter / Space when the portrait has keyboard focus.
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openPortrait();
    }
  }

  if (portrait) {
    portrait.addEventListener("click", openPortrait);
    portrait.addEventListener("keydown", onPortraitActivate);
  }

  } catch (e) {
    // Degrade gracefully: reveal [data-animate] content and surface the error.
    document.documentElement.classList.remove("js");
    console.error("script.js error — falling back to no-JS rendering:", e);
  }
})();
