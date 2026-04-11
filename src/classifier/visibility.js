/**
 * NIOR-AI Five-Phase Visibility Filter (Section V of the paper)
 *
 * Phases (ascending cost order):
 *   1. CSS visibility   – checkVisibility() API
 *   2. Dimension check  – offsetWidth/offsetHeight
 *   3. Viewport intersection – IntersectionObserver
 *   4. Supplementary CSS occlusion – getComputedStyle
 *   5. Hit-test sampling – elementFromPoint
 *
 * Phase 3 is async (IntersectionObserver callback). Phases 1, 2, 4, 5
 * are synchronous and execute in the same rAF callback.
 */

'use strict';

const CLIP_RE = /inset\(.*?(?:calc\()?100%|polygon\(0[^)]*\)/;

/**
 * Phase 1: CSS visibility via checkVisibility (Chrome 105+).
 * Falls back to getComputedStyle on older browsers.
 */
function phase1_cssVisible(el) {
  if (typeof el.checkVisibility === 'function') {
    return el.checkVisibility({
      checkOpacity: true,
      checkVisibilityCSS: true,
      contentVisibilityAuto: true
    });
  }
  // Fallback for Firefox/Safari
  const s = getComputedStyle(el);
  return s.display !== 'none' &&
         s.visibility !== 'hidden' &&
         parseFloat(s.opacity) > 0;
}

/**
 * Phase 2: non-zero rendered dimensions.
 */
function phase2_hasSize(el) {
  return el.offsetWidth > 0 || el.offsetHeight > 0;
}

/**
 * Phase 4: supplementary CSS occlusion.
 * Called only for elements that survived phases 1-3.
 */
function phase4_clipCheck(el) {
  let p = el;
  while (p && p !== document.body) {
    const s = getComputedStyle(p);
    // fully-occluding clip-path
    if (s.clipPath && CLIP_RE.test(s.clipPath)) return false;
    // overflow hidden + zero-size ancestor
    if (s.overflow === 'hidden') {
      const r = p.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) return false;
    }
    p = p.parentElement;
  }
  return true;
}

/**
 * Phase 5: hit-test sampling via elementFromPoint.
 * Tests centroid + 4 inset corners (2px inset).
 * Returns true if any sample point returns the element or a descendant.
 * Must be called inside a rAF callback to avoid forced reflow.
 */
function phase5_hitTest(el, bcr) {
  const points = [
    [bcr.left + bcr.width / 2,  bcr.top + bcr.height / 2],  // centroid
    [bcr.left + 2,              bcr.top + 2],                 // top-left
    [bcr.right - 2,             bcr.top + 2],                 // top-right
    [bcr.left + 2,              bcr.bottom - 2],              // bottom-left
    [bcr.right - 2,             bcr.bottom - 2]               // bottom-right
  ];
  for (const [x, y] of points) {
    const hit = document.elementFromPoint(x, y);
    if (hit && (hit === el || el.contains(hit))) return true;
  }
  return false;
}

/**
 * Apply phases 1, 2, 4, and 5 to a pre-collected candidate list.
 * Phase 3 (IntersectionObserver) is handled separately in CandidateCollector
 * because it is inherently asynchronous.
 *
 * Phase 5 (hit-test) is deferred to a rAF callback to avoid forced reflow,
 * so this function returns a Promise.
 *
 * @param {Array<{element: Element}>} candidates
 * @returns {Promise<Array<{element: Element, bcr: DOMRect}>>} visible subset with BCR
 */
export function syncVisibilityFilter(candidates) {
  const p1 = candidates.filter(({ element }) => phase1_cssVisible(element));
  const p2 = p1.filter(({ element }) => phase2_hasSize(element));
  const p4 = p2.filter(({ element }) => phase4_clipCheck(element));

  // Phase 5 is batched: all elementFromPoint calls happen together
  // inside one rAF to prevent multiple forced reflows.
  return new Promise(resolve => {
    requestAnimationFrame(() => {
      // Batch BCR reads first (no writes)
      const withBCR = p4.map(({ element }) => ({
        element,
        bcr: element.getBoundingClientRect()
      }));
      // Then hit-test
      const visible = withBCR.filter(({ element, bcr }) =>
        bcr.width > 0 && bcr.height > 0 && phase5_hitTest(element, bcr)
      );
      resolve(visible);
    });
  });
}

/**
 * CandidateCollector
 *
 * Manages the three-gate triggering and scheduling pipeline
 * (Section VI-F of the paper):
 *   Gate 1 – MutationObserver significance threshold (N_min = 5)
 *   Gate 2 – IntersectionObserver viewport filter (rootMargin 200px)
 *   Gate 3 – 300ms debounce + requestIdleCallback dispatch
 */
export class CandidateCollector {
  constructor({ onBatch, nMin = 5, debounceMs = 300, batchSize = 20 }) {
    this._onBatch    = onBatch;
    this._nMin       = nMin;
    this._debounceMs = debounceMs;
    this._batchSize  = batchSize;
    this._pending    = new Set();
    this._seen       = new WeakSet();
    this._timer      = null;

    this._io = new IntersectionObserver(
      this._onIntersect.bind(this),
      { rootMargin: '200px', threshold: 0.1 }
    );

    this._mo = new MutationObserver(this._onMutation.bind(this));
    this._mo.observe(document.body, {
      childList: true,
      subtree: true
    });

    // SPA navigation detection
    window.addEventListener('popstate',   () => this._onNavigation(), { passive: true });
    window.addEventListener('hashchange', () => this._onNavigation(), { passive: true });
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
        { timeout: 2000 }
      );
    }, this._debounceMs);
  }

  _dispatchInBatches(elements) {
    for (let i = 0; i < elements.length; i += this._batchSize) {
      this._onBatch(elements.slice(i, i + this._batchSize));
    }
  }

  _onNavigation() {
    // Reset seen-set and schedule a full re-scan after navigation settles
    this._seen = new WeakSet();
    this._pending.clear();
    clearTimeout(this._timer);
    setTimeout(() => {
      const all = Array.from(document.querySelectorAll('*'));
      this._dispatchInBatches(all);
    }, 500);
  }

  destroy() {
    this._mo.disconnect();
    this._io.disconnect();
    clearTimeout(this._timer);
  }
}
