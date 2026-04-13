/**
 * NIOR-AI Shadow Renderer (Section IV of the paper)
 *
 * Hosts the annotation canvas inside a closed Shadow DOM root so that:
 *   - The host page's CSS selectors cannot reach it
 *   - querySelector / getElement* return null for shadow-internal nodes
 *   - MutationObserver callbacks on the light tree never fire for shadow mutations
 *   - React/Vue/Angular reconciliation cycles ignore it entirely
 *
 * The canvas is GPU-promoted via will-change:transform and position:fixed,
 * providing an identity coordinate mapping: annotation (x,y) = BCR (x,y).
 */

'use strict';

import { MaskedRedrawScheduler } from './masked-redraw.js';

const LABEL_COLORS = {
  main_content:    '#2563EB',
  navigation:      '#16A34A',
  call_to_action:  '#DC2626',
  form:            '#9333EA',
  heading:         '#0891B2',
  sidebar:         '#D97706',
  media:           '#DB2777',
  advertisement:   '#64748B'
};

const CANVAS_STYLES = `
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

/**
 * ShadowRenderer
 *
 * Manages the closed shadow root, canvas, and the collection of live
 * annotation handles. Delegates frame scheduling and dirty-region
 * computation to MaskedRedrawScheduler.
 */
export class ShadowRenderer {
  constructor() {
    this._annotations = new Map(); // handle → AnnotationRecord
    this._handleSeq   = 0;
    this._batchMode   = false;

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
    // Create a dedicated host element so body's light-DOM children remain
    // visible. Attaching shadow directly to <body> suppresses rendering of
    // all body children (no <slot>), making the page content disappear.
    const host = document.createElement('div');
    host.style.cssText = 'all:initial;position:fixed;top:0;left:0;width:0;height:0;overflow:visible;pointer-events:none;';
    document.documentElement.appendChild(host);

    // Closed shadow root: element.shadowRoot returns null externally
    this._shadowRoot = host.attachShadow({ mode: 'closed' });

    // Style encapsulation
    const style = document.createElement('style');
    style.textContent = CANVAS_STYLES;
    this._shadowRoot.appendChild(style);

    // Single canvas covers the entire viewport
    this._canvas = document.createElement('canvas');
    this._shadowRoot.appendChild(this._canvas);

    this._ctx = this._canvas.getContext('2d');
    this._setCanvasSize();  // must be called after _ctx is set so scale(dpr,dpr) applies
  }

  _setCanvasSize() {
    const dpr = window.devicePixelRatio || 1;
    this._canvas.width  = window.innerWidth  * dpr;
    this._canvas.height = window.innerHeight * dpr;
    if (this._ctx) this._ctx.scale(dpr, dpr);
  }

  _initObservers() {
    // Observer bridge: marks affected annotations dirty
    this._mo = new MutationObserver(() => this._markAllDirty());
    this._mo.observe(document.body, {
      subtree: true,
      attributeFilter: ['class', 'style', 'hidden']
    });

    this._ro = new ResizeObserver(() => this._onResize());
    this._ro.observe(document.documentElement);

    window.addEventListener('scroll', () => this._markAllDirty(), {
      passive: true,
      capture: true
    });

    window.addEventListener('resize', () => this._onResize(), { passive: true });
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
      config: { type: 'bbox', labelPos: 'top-left', ...config },
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
    // Clear old BCR before deletion.
    // The canvas context already has ctx.scale(dpr, dpr) applied, so BCR
    // values (CSS pixels) map correctly without manual DPR multiplication.
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
    this._shadowRoot.host.shadowRoot; // noop – cannot detach closed root
    this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
  }
}

// ── Drawing helpers (used by MaskedRedrawScheduler) ──────────────────────────

export function drawAnnotation(ctx, record, bcr) {
  const { config } = record;
  const color = config.color || LABEL_COLORS[config.label] || '#2563EB';

  switch (config.type) {
    case 'bbox':      _drawBbox(ctx, bcr, color, config); break;
    case 'highlight': _drawHighlight(ctx, bcr, color);    break;
    case 'label':     _drawLabel(ctx, bcr, color, config.label, config.labelPos); break;
    case 'heatmap':   _drawHeatmap(ctx, bcr, color);      break;
  }
}

function _drawBbox(ctx, bcr, color, config) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.strokeRect(bcr.x, bcr.y, bcr.width, bcr.height);
  if (config.fill) {
    ctx.fillStyle = _colorToRgba(color, 0.15);
    ctx.fillRect(bcr.x, bcr.y, bcr.width, bcr.height);
  }
  if (config.label) {
    _drawLabel(ctx, bcr, color, config.label, config.labelPos ?? 'top-left');
  }
  ctx.restore();
}

function _drawHighlight(ctx, bcr, color) {
  ctx.save();
  ctx.globalCompositeOperation = 'multiply';
  ctx.fillStyle = _colorToRgba(color, 0.15);
  ctx.fillRect(bcr.x, bcr.y, bcr.width, bcr.height);
  ctx.restore();
}

/**
 * Heatmap: radial gradient composited with 'screen' so overlapping
 * instances accumulate (Section VIII-B).
 */
function _drawHeatmap(ctx, bcr, color) {
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  const cx = bcr.x + bcr.width  / 2;
  const cy = bcr.y + bcr.height / 2;
  const r  = Math.max(bcr.width, bcr.height) / 2;
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
  // Parse hex/rgb color into rgba for gradient stops
  const rgba = _colorToRgba(color, 0.5);
  grad.addColorStop(0, rgba);
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(bcr.x, bcr.y, bcr.width, bcr.height);
  ctx.restore();
}

/** Convert a hex or rgb color string to rgba(r,g,b,a). */
function _colorToRgba(color, alpha) {
  // Hex #RRGGBB
  const m = color.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
  if (m) return `rgba(${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)},${alpha})`;
  // Already rgb(...)
  if (color.startsWith('rgb(')) return color.replace('rgb(', 'rgba(').replace(')', `,${alpha})`);
  return `rgba(37,99,235,${alpha})`; // fallback blue
}

function _drawLabel(ctx, bcr, color, text, pos) {
  if (!text) return;
  ctx.save();
  ctx.font = 'bold 11px monospace';
  const metrics = ctx.measureText(text);
  const tw = metrics.width + 8;
  const th = 16;

  let lx = bcr.x;
  let ly = bcr.y - th;
  if (pos === 'top-right')    lx = bcr.x + bcr.width - tw;
  if (pos === 'bottom-left')  ly = bcr.y + bcr.height;
  if (pos === 'bottom-right') { lx = bcr.x + bcr.width - tw; ly = bcr.y + bcr.height; }
  if (ly < 0) ly = bcr.y;

  ctx.fillStyle = color;
  ctx.fillRect(lx, ly, tw, th);
  ctx.fillStyle = '#fff';
  ctx.fillText(text, lx + 4, ly + 11);
  ctx.restore();
}
