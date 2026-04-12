"use strict";
(() => {
  // src/classifier/visibility.js
  var CLIP_RE = /inset\(.*?(?:calc\()?100%|polygon\(0[^)]*\)/;
  function phase1_cssVisible(el) {
    if (typeof el.checkVisibility === "function") {
      return el.checkVisibility({
        checkOpacity: true,
        checkVisibilityCSS: true,
        contentVisibilityAuto: true
      });
    }
    const s = getComputedStyle(el);
    return s.display !== "none" && s.visibility !== "hidden" && parseFloat(s.opacity) > 0;
  }
  function phase2_hasSize(el) {
    return el.offsetWidth > 0 || el.offsetHeight > 0;
  }
  function phase4_clipCheck(el) {
    let p = el;
    while (p && p !== document.body) {
      const s = getComputedStyle(p);
      if (s.clipPath && CLIP_RE.test(s.clipPath)) return false;
      if (s.overflow === "hidden") {
        const r = p.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return false;
      }
      p = p.parentElement;
    }
    return true;
  }
  function phase5_hitTest(el, bcr) {
    const points = [
      [bcr.left + bcr.width / 2, bcr.top + bcr.height / 2],
      // centroid
      [bcr.left + 2, bcr.top + 2],
      // top-left
      [bcr.right - 2, bcr.top + 2],
      // top-right
      [bcr.left + 2, bcr.bottom - 2],
      // bottom-left
      [bcr.right - 2, bcr.bottom - 2]
      // bottom-right
    ];
    for (const [x, y] of points) {
      const hit = document.elementFromPoint(x, y);
      if (hit && (hit === el || el.contains(hit))) return true;
    }
    return false;
  }
  function syncVisibilityFilter(candidates) {
    const p1 = candidates.filter(({ element }) => phase1_cssVisible(element));
    const p2 = p1.filter(({ element }) => phase2_hasSize(element));
    const p4 = p2.filter(({ element }) => phase4_clipCheck(element));
    return new Promise((resolve) => {
      requestAnimationFrame(() => {
        const withBCR = p4.map(({ element }) => ({
          element,
          bcr: element.getBoundingClientRect()
        }));
        const visible = withBCR.filter(
          ({ element, bcr }) => bcr.width > 0 && bcr.height > 0 && phase5_hitTest(element, bcr)
        );
        resolve(visible);
      });
    });
  }
  var CandidateCollector = class {
    constructor({ onBatch, nMin = 5, debounceMs = 300, batchSize = 20 }) {
      this._onBatch = onBatch;
      this._nMin = nMin;
      this._debounceMs = debounceMs;
      this._batchSize = batchSize;
      this._pending = /* @__PURE__ */ new Set();
      this._seen = /* @__PURE__ */ new WeakSet();
      this._timer = null;
      this._io = new IntersectionObserver(
        this._onIntersect.bind(this),
        { rootMargin: "200px", threshold: 0.1 }
      );
      this._mo = new MutationObserver(this._onMutation.bind(this));
      this._mo.observe(document.body, {
        childList: true,
        subtree: true
      });
      window.addEventListener("popstate", () => this._onNavigation(), { passive: true });
      window.addEventListener("hashchange", () => this._onNavigation(), { passive: true });
    }
    // ── Gate 1 ────────────────────────────────────────────────────────────────
    _onMutation(records) {
      const newElements = [];
      for (const r of records) {
        for (const node of r.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) newElements.push(node);
        }
      }
      if (newElements.length < this._nMin) return;
      for (const el of newElements) {
        if (!this._seen.has(el)) this._io.observe(el);
      }
    }
    // ── Gate 2 ────────────────────────────────────────────────────────────────
    _onIntersect(entries) {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          this._io.unobserve(entry.target);
          if (!this._seen.has(entry.target)) {
            this._pending.add(entry.target);
            this._scheduleDispatch();
          }
        }
      }
    }
    // ── Gate 3 ────────────────────────────────────────────────────────────────
    _scheduleDispatch() {
      clearTimeout(this._timer);
      this._timer = setTimeout(() => {
        const elements = [...this._pending];
        this._pending.clear();
        for (const el of elements) this._seen.add(el);
        requestIdleCallback(
          () => this._dispatchInBatches(elements),
          { timeout: 2e3 }
        );
      }, this._debounceMs);
    }
    _dispatchInBatches(elements) {
      for (let i = 0; i < elements.length; i += this._batchSize) {
        this._onBatch(elements.slice(i, i + this._batchSize));
      }
    }
    _onNavigation() {
      this._seen = /* @__PURE__ */ new WeakSet();
      this._pending.clear();
      clearTimeout(this._timer);
      setTimeout(() => {
        const all = Array.from(document.querySelectorAll("*"));
        this._dispatchInBatches(all);
      }, 500);
    }
    destroy() {
      this._mo.disconnect();
      this._io.disconnect();
      clearTimeout(this._timer);
    }
  };

  // src/classifier/features.js
  var TAG_INDEX = {
    nav: 0,
    header: 1,
    footer: 2,
    main: 3,
    aside: 4,
    section: 5,
    article: 6,
    form: 7,
    button: 8,
    input: 9,
    select: 10,
    a: 11,
    h: 12,
    // h1-h6 collapsed
    other: 13
  };
  var ARIA_INDEX = {
    navigation: 0,
    main: 1,
    banner: 2,
    contentinfo: 3,
    complementary: 4,
    form: 5,
    search: 6
  };
  var KEYWORDS = [
    "nav",
    "navigation",
    "sidebar",
    "side-bar",
    "hero",
    "cta",
    "footer",
    "header",
    "form",
    "ad",
    "advertisement",
    "banner",
    "content",
    "main",
    "modal",
    "popup",
    "menu",
    "dropdown",
    "carousel",
    "slider",
    "widget",
    "card",
    "article",
    "post",
    "gallery",
    "media",
    "video",
    "image",
    "search",
    "login",
    "signup",
    "checkout"
  ];
  var MEDIAN_DEPTH_CACHE = /* @__PURE__ */ new WeakMap();
  function getMedianDepth(doc) {
    if (MEDIAN_DEPTH_CACHE.has(doc)) return MEDIAN_DEPTH_CACHE.get(doc);
    const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT);
    const sample = [];
    let node;
    let count = 0;
    while ((node = walker.nextNode()) !== null) {
      let depth = 0;
      let p = node;
      while (p.parentElement) {
        depth++;
        p = p.parentElement;
      }
      count++;
      if (sample.length < 200) {
        sample.push(depth);
      } else {
        const j = Math.floor(Math.random() * count);
        if (j < 200) sample[j] = depth;
      }
    }
    const sorted = sample.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] || 1;
    MEDIAN_DEPTH_CACHE.set(doc, median);
    return median;
  }
  function elementDepth(el) {
    let d = 0;
    let p = el;
    while (p.parentElement) {
      d++;
      p = p.parentElement;
    }
    return d;
  }
  function ancestorBlockArea(el, vw, vh) {
    const BLOCK_TAGS = /* @__PURE__ */ new Set([
      "DIV",
      "SECTION",
      "ARTICLE",
      "ASIDE",
      "MAIN",
      "HEADER",
      "FOOTER",
      "NAV",
      "FORM",
      "LI",
      "TD",
      "TH"
    ]);
    let p = el.parentElement;
    while (p) {
      if (BLOCK_TAGS.has(p.tagName) && typeof p.getBoundingClientRect === "function") {
        const r = p.getBoundingClientRect();
        const area = r.width * r.height;
        if (area > 0) return area;
      }
      p = p.parentElement;
    }
    return vw * vh;
  }
  function extractFeatures(el, bcr, vw, vh, scrollH, medDepth, scrollY = 0) {
    const vec = new Float32Array(72);
    const tag = el.tagName.toLowerCase();
    const tagKey = /^h[1-6]$/.test(tag) ? "h" : TAG_INDEX[tag] !== void 0 ? tag : "other";
    vec[TAG_INDEX[tagKey] ?? 13] = 1;
    const role = (el.getAttribute("role") || "").toLowerCase();
    const ariaIdx = ARIA_INDEX[role];
    if (ariaIdx !== void 0) vec[14 + ariaIdx] = 1;
    vec[21] = medDepth > 0 ? Math.min(elementDepth(el) / medDepth, 3) : 0;
    const area = bcr.width * bcr.height;
    vec[22] = bcr.left / vw;
    vec[23] = bcr.top / vh;
    vec[24] = bcr.width / vw;
    vec[25] = bcr.height / vh;
    vec[26] = vw * vh > 0 ? area / (vw * vh) : 0;
    const pageTop = bcr.top + scrollY;
    vec[27] = scrollH > 0 ? pageTop / scrollH : 0;
    vec[28] = bcr.top >= 0 && bcr.top < vh ? 1 : 0;
    const ancestorArea = ancestorBlockArea(el, vw, vh);
    vec[29] = ancestorArea > 0 ? area / ancestorArea : 0;
    const text = el.innerText || "";
    const words = text.trim().split(/\s+/).filter(Boolean);
    const links = el.querySelectorAll("a");
    const imgs = el.querySelectorAll("img, video, audio, canvas, svg");
    const interactives = el.querySelectorAll(
      'button, input, select, textarea, [role="button"], [tabindex]'
    );
    const children = el.children.length;
    const descendants = el.querySelectorAll("*").length;
    vec[30] = text.length > 0 ? Math.log1p(text.length) : 0;
    vec[31] = words.length > 0 ? Math.log1p(words.length) : 0;
    vec[32] = Math.log1p(links.length);
    vec[33] = words.length > 0 ? links.length / words.length : 0;
    vec[34] = Math.log1p(imgs.length);
    vec[35] = Math.log1p(interactives.length);
    vec[36] = Math.log1p(children);
    vec[37] = Math.log1p(descendants);
    const tokens = new Set(
      ((el.className || "") + " " + (el.id || "")).toLowerCase().split(/[\s\-_]+/).filter(Boolean)
    );
    for (let i = 0; i < KEYWORDS.length; i++) {
      vec[38 + i] = tokens.has(KEYWORDS[i]) ? 1 : 0;
    }
    vec[70] = 0;
    vec[71] = 0;
    return vec;
  }
  function batchExtract(candidates) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const scrollH = document.documentElement.scrollHeight;
    const scrollY = window.scrollY || 0;
    const medDepth = getMedianDepth(document);
    const pairs = candidates.map(({ element, bcr }) => ({
      element,
      bcr: bcr ?? element.getBoundingClientRect()
    }));
    return pairs.map(({ element, bcr }) => ({
      element,
      features: extractFeatures(element, bcr, vw, vh, scrollH, medDepth, scrollY)
    }));
  }

  // src/shadow-renderer/masked-redraw.js
  var MaskedRedrawScheduler = class {
    /**
     * @param {CanvasRenderingContext2D} ctx
     * @param {Map<number, AnnotationRecord>} annotations  – shared reference
     * @param {() => {w: number, h: number}} getSize
     */
    constructor(ctx, annotations, getSize) {
      this._ctx = ctx;
      this._annotations = annotations;
      this._getSize = getSize;
      this._rafId = null;
    }
    requestFrame() {
      if (this._rafId !== null) return;
      this._rafId = requestAnimationFrame(() => {
        this._rafId = null;
        this.flush();
      });
    }
    /**
     * Synchronous flush – executes all four algorithm phases.
     * Safe to call from within a rAF callback.
     */
    flush() {
      const dirty = [...this._annotations.values()].filter((r) => r.dirty);
      if (dirty.length === 0) return;
      const { w, h } = this._getSize();
      const updates = dirty.map((record) => ({
        record,
        bcrNew: record.element.getBoundingClientRect()
      }));
      const dirtyRects = [];
      for (const { record, bcrNew } of updates) {
        if (record.bcrOld) dirtyRects.push(record.bcrOld);
        if (bcrNew.width > 0 && bcrNew.height > 0) dirtyRects.push(bcrNew);
      }
      if (dirtyRects.length === 0) {
        for (const { record } of updates) record.dirty = false;
        return;
      }
      const bcrNewByRecord = new Map(
        updates.map(({ record, bcrNew }) => [record, bcrNew])
      );
      const ctx = this._ctx;
      ctx.save();
      ctx.beginPath();
      for (const r of dirtyRects) {
        ctx.rect(r.x, r.y, r.width, r.height);
      }
      ctx.clip();
      ctx.clearRect(0, 0, w, h);
      for (const record of this._annotations.values()) {
        const bcr = bcrNewByRecord.get(record) ?? record.bcrOld;
        if (bcr && bcr.width > 0 && bcr.height > 0 && _intersectsAny(bcr, dirtyRects)) {
          drawAnnotation(ctx, record, bcr);
        }
      }
      ctx.restore();
      for (const { record, bcrNew } of updates) {
        record.bcrOld = bcrNew.width > 0 ? bcrNew : null;
        record.dirty = false;
      }
    }
    destroy() {
      if (this._rafId !== null) {
        cancelAnimationFrame(this._rafId);
        this._rafId = null;
      }
    }
  };
  function _intersectsAny(a, rects) {
    for (const b of rects) {
      if (a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y) {
        return true;
      }
    }
    return false;
  }

  // src/shadow-renderer/renderer.js
  var LABEL_COLORS = {
    main_content: "#2563EB",
    navigation: "#16A34A",
    call_to_action: "#DC2626",
    form: "#9333EA",
    heading: "#0891B2",
    sidebar: "#D97706",
    media: "#DB2777",
    advertisement: "#64748B"
  };
  var CANVAS_STYLES = `
  canvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    will-change: transform;
    z-index: 2147483647;
  }
`;
  var ShadowRenderer = class {
    constructor() {
      this._annotations = /* @__PURE__ */ new Map();
      this._handleSeq = 0;
      this._batchMode = false;
      this._initShadow();
      this._initObservers();
      this._resizeScheduled = false;
      this._scheduler = new MaskedRedrawScheduler(
        this._ctx,
        this._annotations,
        () => ({ w: this._canvas.width, h: this._canvas.height })
      );
    }
    // ── Initialisation ────────────────────────────────────────────────────────
    _initShadow() {
      const host = document.body;
      this._shadowRoot = host.attachShadow({ mode: "closed" });
      const style = document.createElement("style");
      style.textContent = CANVAS_STYLES;
      this._shadowRoot.appendChild(style);
      this._canvas = document.createElement("canvas");
      this._setCanvasSize();
      this._shadowRoot.appendChild(this._canvas);
      this._ctx = this._canvas.getContext("2d");
    }
    _setCanvasSize() {
      const dpr = window.devicePixelRatio || 1;
      this._canvas.width = window.innerWidth * dpr;
      this._canvas.height = window.innerHeight * dpr;
      if (this._ctx) this._ctx.scale(dpr, dpr);
    }
    _initObservers() {
      this._mo = new MutationObserver(() => this._markAllDirty());
      this._mo.observe(document.body, {
        subtree: true,
        attributeFilter: ["class", "style", "hidden"]
      });
      this._ro = new ResizeObserver(() => this._onResize());
      this._ro.observe(document.documentElement);
      window.addEventListener("scroll", () => this._markAllDirty(), {
        passive: true,
        capture: true
      });
      window.addEventListener("resize", () => this._onResize(), { passive: true });
    }
    _onResize() {
      if (this._resizeScheduled) return;
      this._resizeScheduled = true;
      requestAnimationFrame(() => {
        this._resizeScheduled = false;
        this._setCanvasSize();
        this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
        for (const record of this._annotations.values()) {
          record.dirty = true;
        }
        this._scheduler.flush();
      });
    }
    _markAllDirty() {
      for (const record of this._annotations.values()) {
        record.dirty = true;
      }
      this._scheduler.requestFrame();
    }
    // ── Public Annotation API ─────────────────────────────────────────────────
    /**
     * Annotate an element. Returns a handle for update/remove.
     *
     * @param {Element} element
     * @param {{ type: 'bbox'|'highlight'|'label', color: string,
     *            label?: string, labelPos?: string }} config
     * @returns {number} annotation handle
     */
    annotate(element, config) {
      const handle = ++this._handleSeq;
      const record = {
        element,
        config: { type: "bbox", labelPos: "top-left", ...config },
        bcrOld: null,
        dirty: true
      };
      this._annotations.set(handle, record);
      if (!this._batchMode) {
        this._scheduler.requestFrame();
      }
      return handle;
    }
    /**
     * Update annotation config for an existing handle.
     */
    update(handle, config) {
      const record = this._annotations.get(handle);
      if (!record) return;
      Object.assign(record.config, config);
      record.dirty = true;
      if (!this._batchMode) this._scheduler.requestFrame();
    }
    /**
     * Remove an annotation by handle.
     */
    remove(handle) {
      const record = this._annotations.get(handle);
      if (!record) return;
      if (record.bcrOld) {
        const { x, y, width, height } = record.bcrOld;
        this._ctx.clearRect(x, y, width, height);
      }
      this._annotations.delete(handle);
    }
    /**
     * Batch multiple annotate/update calls into a single rAF flush.
     * Prevents partial-frame flicker when classifying many elements.
     */
    batch(fn) {
      this._batchMode = true;
      try {
        fn();
      } finally {
        this._batchMode = false;
        this._scheduler.requestFrame();
      }
    }
    /**
     * Remove all annotations and clear the canvas.
     */
    clear() {
      this._annotations.clear();
      this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    }
    /**
     * Tear down observers and remove the shadow host.
     */
    destroy() {
      this._mo.disconnect();
      this._ro.disconnect();
      this._scheduler.destroy();
      this._shadowRoot.host.shadowRoot;
      this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    }
  };
  function drawAnnotation(ctx, record, bcr) {
    const { config } = record;
    const color = config.color || LABEL_COLORS[config.label] || "#2563EB";
    switch (config.type) {
      case "bbox":
        _drawBbox(ctx, bcr, color, config);
        break;
      case "highlight":
        _drawHighlight(ctx, bcr, color);
        break;
      case "label":
        _drawLabel(ctx, bcr, color, config.label, config.labelPos);
        break;
      case "heatmap":
        _drawHeatmap(ctx, bcr, color);
        break;
    }
  }
  function _drawBbox(ctx, bcr, color, config) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(bcr.x, bcr.y, bcr.width, bcr.height);
    if (config.fill) {
      ctx.fillStyle = color.replace(")", ", 0.08)").replace("rgb", "rgba");
      ctx.fillRect(bcr.x, bcr.y, bcr.width, bcr.height);
    }
    if (config.label) {
      _drawLabel(ctx, bcr, color, config.label, config.labelPos ?? "top-left");
    }
    ctx.restore();
  }
  function _drawHighlight(ctx, bcr, color) {
    ctx.save();
    ctx.globalCompositeOperation = "multiply";
    ctx.fillStyle = color.replace(")", ", 0.15)").replace("rgb", "rgba") || "#2563EB33";
    ctx.fillRect(bcr.x, bcr.y, bcr.width, bcr.height);
    ctx.restore();
  }
  function _drawHeatmap(ctx, bcr, color) {
    ctx.save();
    ctx.globalCompositeOperation = "screen";
    const cx = bcr.x + bcr.width / 2;
    const cy = bcr.y + bcr.height / 2;
    const r = Math.max(bcr.width, bcr.height) / 2;
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    const rgba = _colorToRgba(color, 0.5);
    grad.addColorStop(0, rgba);
    grad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(bcr.x, bcr.y, bcr.width, bcr.height);
    ctx.restore();
  }
  function _colorToRgba(color, alpha) {
    const m = color.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    if (m) return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${alpha})`;
    if (color.startsWith("rgb(")) return color.replace("rgb(", "rgba(").replace(")", `,${alpha})`);
    return `rgba(37,99,235,${alpha})`;
  }
  function _drawLabel(ctx, bcr, color, text, pos) {
    if (!text) return;
    ctx.save();
    ctx.font = "bold 11px monospace";
    const metrics = ctx.measureText(text);
    const tw = metrics.width + 8;
    const th = 16;
    let lx = bcr.x;
    let ly = bcr.y - th;
    if (pos === "top-right") lx = bcr.x + bcr.width - tw;
    if (pos === "bottom-left") ly = bcr.y + bcr.height;
    if (pos === "bottom-right") {
      lx = bcr.x + bcr.width - tw;
      ly = bcr.y + bcr.height;
    }
    if (ly < 0) ly = bcr.y;
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, lx + 4, ly + 11);
    ctx.restore();
  }

  // src/content/content.js
  var renderer = null;
  var collector = null;
  var elementHandles = /* @__PURE__ */ new Map();
  function init() {
    renderer = new ShadowRenderer();
    collector = new CandidateCollector({
      onBatch: handleCandidateBatch,
      nMin: 5,
      debounceMs: 300,
      batchSize: 20
    });
    chrome.runtime.onMessage.addListener(handleMessage);
  }
  async function handleCandidateBatch(elements) {
    const visible = await syncVisibilityFilter(
      elements.map((el) => ({ element: el }))
    );
    if (visible.length === 0) return;
    const featureData = batchExtract(visible);
    const metas = featureData.map(({ element }) => ({
      tag: element.tagName.toLowerCase(),
      ariaRole: element.getAttribute("role") || ""
    }));
    chrome.runtime.sendMessage({
      type: "CLASSIFY_REQUEST",
      batch: featureData.map(({ features }) => Array.from(features)),
      metas,
      // Include element indices so we can correlate results
      elementIds: featureData.map((_, i) => i)
    }, (response) => {
      if (chrome.runtime.lastError || !response) return;
      applyPredictions(featureData.map((d) => d.element), response.predictions);
    });
  }
  function applyPredictions(elements, predictions) {
    renderer.batch(() => {
      elements.forEach((el, i) => {
        const { label, confidence } = predictions[i];
        const existing = elementHandles.get(el);
        if (existing !== void 0) renderer.remove(existing);
        const handle = renderer.annotate(el, {
          type: "bbox",
          label,
          color: labelColor(label),
          labelPos: "top-left",
          fill: confidence > 0.8
        });
        elementHandles.set(el, handle);
      });
    });
  }
  function handleMessage(msg) {
    if (msg.type === "NIOR_CLEAR") {
      renderer?.clear();
      elementHandles.clear();
    }
    if (msg.type === "NIOR_TOGGLE") {
      toggleOverlay(msg.visible);
    }
  }
  function toggleOverlay(visible) {
    const canvas = renderer?._canvas;
    if (canvas) canvas.style.display = visible ? "" : "none";
  }
  var LABEL_COLORS2 = {
    main_content: "#2563EB",
    navigation: "#16A34A",
    call_to_action: "#DC2626",
    form: "#9333EA",
    heading: "#0891B2",
    sidebar: "#D97706",
    media: "#DB2777",
    advertisement: "#64748B"
  };
  function labelColor(label) {
    return LABEL_COLORS2[label] ?? "#6B7280";
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
