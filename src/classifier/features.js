/**
 * NIOR-AI Feature Extractor
 *
 * Extracts the 72-dimensional feature vector described in Section VI-B
 * of the paper. All DOM reads are batched before the rAF boundary to
 * prevent layout thrashing.
 *
 * Feature groups:
 *   [0..21]  Structural features   (22 dims)
 *              [0..13]  14-way tag one-hot
 *              [14..20]  7-way ARIA role one-hot (unrecognised role → all zero)
 *              [21]      DOM depth normalised by median tree depth
 *   [22..29] Geometric features    ( 8 dims)
 *   [30..71] Content features      (42 dims)
 *              [30..37]  8 log-normalised content counts
 *              [38..69]  32-dim keyword indicator
 *              [70..71]  reserved / padding
 */

'use strict';

// ── Structural: 14-way tag one-hot ──────────────────────────────────────────
const TAG_INDEX = {
  nav: 0, header: 1, footer: 2, main: 3, aside: 4,
  section: 5, article: 6, form: 7, button: 8, input: 9,
  select: 10, a: 11, h: 12,  // h1-h6 collapsed
  other: 13
};

// ── Structural: 7-way ARIA role one-hot [dims 14..20] ───────────────────────
// 'none' and any unrecognised role leave all 7 bits zero (implicit none).
// Dim 21 is reserved exclusively for normalised DOM depth.
const ARIA_INDEX = {
  navigation: 0, main: 1, banner: 2, contentinfo: 3,
  complementary: 4, form: 5, search: 6
};

// ── Content: 32 most discriminative class/id keyword tokens ─────────────────
const KEYWORDS = [
  'nav', 'navigation', 'sidebar', 'side-bar', 'hero', 'cta',
  'footer', 'header', 'form', 'ad', 'advertisement', 'banner',
  'content', 'main', 'modal', 'popup', 'menu', 'dropdown',
  'carousel', 'slider', 'widget', 'card', 'article', 'post',
  'gallery', 'media', 'video', 'image', 'search', 'login',
  'signup', 'checkout'
];

const MEDIAN_DEPTH_CACHE = new WeakMap();

/**
 * Returns the median DOM depth of the document, cached per document object.
 * Uses reservoir sampling (k=200) to avoid full traversal on large DOMs.
 */
function getMedianDepth(doc) {
  if (MEDIAN_DEPTH_CACHE.has(doc)) return MEDIAN_DEPTH_CACHE.get(doc);

  const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT);
  const sample = [];
  let node;
  let count = 0;

  while ((node = walker.nextNode()) !== null) {
    let depth = 0;
    let p = node;
    while (p.parentElement) { depth++; p = p.parentElement; }
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

/**
 * Compute the depth of a single element from the document root.
 */
function elementDepth(el) {
  let d = 0;
  let p = el;
  while (p.parentElement) { d++; p = p.parentElement; }
  return d;
}

/**
 * Returns the nearest block-level ancestor area (px²), or viewport area
 * as fallback.
 */
function ancestorBlockArea(el, vw, vh) {
  const BLOCK_TAGS = new Set([
    'DIV', 'SECTION', 'ARTICLE', 'ASIDE', 'MAIN',
    'HEADER', 'FOOTER', 'NAV', 'FORM', 'LI', 'TD', 'TH'
  ]);
  let p = el.parentElement;
  while (p) {
    if (BLOCK_TAGS.has(p.tagName) && typeof p.getBoundingClientRect === 'function') {
      const r = p.getBoundingClientRect();
      const area = r.width * r.height;
      if (area > 0) return area;
    }
    p = p.parentElement;
  }
  return vw * vh;
}

/**
 * Extract the 72-dimensional feature vector for one element.
 *
 * @param {Element} el        - Target element (already visibility-filtered)
 * @param {DOMRect} bcr       - Pre-fetched getBoundingClientRect result
 * @param {number}  vw        - Viewport width  (window.innerWidth)
 * @param {number}  vh        - Viewport height (window.innerHeight)
 * @param {number}  scrollH   - document.documentElement.scrollHeight
 * @param {number}  medDepth  - Median tree depth (from getMedianDepth)
 * @param {number}  [scrollY=0] - window.scrollY (passed explicitly for testability)
 * @returns {Float32Array}    - 72-element feature vector
 */
export function extractFeatures(el, bcr, vw, vh, scrollH, medDepth, scrollY = 0) {
  const vec = new Float32Array(72);

  // ── Structural: tag one-hot [0..13] ───────────────────────────────────────
  const tag = el.tagName.toLowerCase();
  const tagKey = /^h[1-6]$/.test(tag) ? 'h' : (TAG_INDEX[tag] !== undefined ? tag : 'other');
  vec[TAG_INDEX[tagKey] ?? 13] = 1;

  // ── Structural: ARIA role one-hot [14..20] ────────────────────────────────
  // Unrecognised role (including 'none') leaves all 7 bits zero.
  const role = (el.getAttribute('role') || '').toLowerCase();
  const ariaIdx = ARIA_INDEX[role];
  if (ariaIdx !== undefined) vec[14 + ariaIdx] = 1;

  // ── Structural: normalised DOM depth [dim 21] ─────────────────────────────
  // Dim 21 is dedicated to depth and never aliases an ARIA bit.
  vec[21] = medDepth > 0 ? Math.min(elementDepth(el) / medDepth, 3) : 0;

  // ── Geometric features [22..29] ───────────────────────────────────────────
  const area = bcr.width * bcr.height;
  vec[22] = bcr.left / vw;                       // normalised x
  vec[23] = bcr.top  / vh;                       // normalised y
  vec[24] = bcr.width  / vw;                     // normalised width
  vec[25] = bcr.height / vh;                     // normalised height
  vec[26] = vw * vh > 0 ? area / (vw * vh) : 0; // fraction of viewport
  // Relative vertical position in scroll range
  const pageTop = bcr.top + scrollY;
  vec[27] = scrollH > 0 ? pageTop / scrollH : 0;
  vec[28] = bcr.top >= 0 && bcr.top < vh ? 1 : 0; // above-the-fold flag
  const ancestorArea = ancestorBlockArea(el, vw, vh);
  vec[29] = ancestorArea > 0 ? area / ancestorArea : 0;

  // ── Content features [30..71] ─────────────────────────────────────────────
  const text = el.innerText || '';
  const words = text.trim().split(/\s+/).filter(Boolean);
  const links = el.querySelectorAll('a');
  const imgs  = el.querySelectorAll('img, video, audio, canvas, svg');
  const interactives = el.querySelectorAll(
    'button, input, select, textarea, [role="button"], [tabindex]'
  );
  const children    = el.children.length;
  const descendants = el.querySelectorAll('*').length;

  vec[30] = text.length > 0 ? Math.log1p(text.length) : 0;  // log char count
  vec[31] = words.length > 0 ? Math.log1p(words.length) : 0; // log word count
  vec[32] = Math.log1p(links.length);                        // log link count
  vec[33] = words.length > 0 ? links.length / words.length : 0; // link-text ratio
  vec[34] = Math.log1p(imgs.length);                         // log media count
  vec[35] = Math.log1p(interactives.length);                 // interactive count
  vec[36] = Math.log1p(children);                            // child count
  vec[37] = Math.log1p(descendants);                         // descendant count

  // 32-dim keyword indicator [38..69]
  const tokens = new Set(
    ((el.className || '') + ' ' + (el.id || ''))
      .toLowerCase()
      .split(/[\s\-_]+/)
      .filter(Boolean)
  );
  for (let i = 0; i < KEYWORDS.length; i++) {
    vec[38 + i] = tokens.has(KEYWORDS[i]) ? 1 : 0;
  }

  // dims 70, 71: padding / reserved
  vec[70] = 0;
  vec[71] = 0;

  return vec;
}

/**
 * Batch-extract features for an array of {element, bcr} pairs.
 * Separates read phase (BCR) from compute phase to avoid layout thrashing.
 *
 * @param {Array<{element: Element, bcr?: DOMRect}>} candidates
 * @returns {Array<{element: Element, features: Float32Array}>}
 */
export function batchExtract(candidates) {
  const vw      = window.innerWidth;
  const vh      = window.innerHeight;
  const scrollH = document.documentElement.scrollHeight;
  const scrollY = window.scrollY || 0;
  const medDepth = getMedianDepth(document);

  // Phase 1: batch all BCR reads
  const pairs = candidates.map(({ element, bcr }) => ({
    element,
    bcr: bcr ?? element.getBoundingClientRect()
  }));

  // Phase 2: compute features (no DOM writes)
  return pairs.map(({ element, bcr }) => ({
    element,
    features: extractFeatures(element, bcr, vw, vh, scrollH, medDepth, scrollY)
  }));
}
