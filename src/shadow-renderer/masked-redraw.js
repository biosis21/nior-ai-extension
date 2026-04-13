/**
 * NIOR-AI Masked Redraw Scheduler
 *
 * Implements Algorithm 1 from Section VII of the paper:
 *
 *   Phase 1 – batch BCR reads  (no writes, avoids layout thrashing)
 *   Phase 2 – compute dirty region  (union of old + new BCRs)
 *   Phase 3 – clip → clearRect → redraw only dirty region
 *   Phase 4 – commit BCR cache, clear dirty flags
 *
 * Correctness guarantees proved in the paper:
 *   • No ghost artefacts  (Proposition 2)
 *   • No overdraw outside dirty region  (Proposition 3)
 *   • Sub-frame completion with prob ≥ 0.98  (Theorem 1)
 */

'use strict';

import { drawAnnotation } from './renderer.js';

export class MaskedRedrawScheduler {
  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {Map<number, AnnotationRecord>} annotations  – shared reference
   * @param {() => {w: number, h: number}} getSize
   */
  constructor(ctx, annotations, getSize) {
    this._ctx         = ctx;
    this._annotations = annotations;
    this._getSize     = getSize;
    this._rafId       = null;
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
    const dirty = [...this._annotations.values()].filter(r => r.dirty);
    if (dirty.length === 0) return;

    // getSize() returns physical pixel dimensions (canvas.width/height).
    // clearRect must use CSS-pixel coordinates because ctx has scale(dpr,dpr).
    const { w: pw, h: ph } = this._getSize();
    const dpr = window.devicePixelRatio || 1;
    const w = pw / dpr;
    const h = ph / dpr;

    // ── Phase 1: batch BCR reads ──────────────────────────────────────────
    const updates = dirty.map(record => ({
      record,
      bcrNew: record.element.getBoundingClientRect()
    }));

    // ── Phase 2: compute dirty region ─────────────────────────────────────
    const dirtyRects = [];
    for (const { record, bcrNew } of updates) {
      if (record.bcrOld) dirtyRects.push(record.bcrOld);
      if (bcrNew.width > 0 && bcrNew.height > 0) dirtyRects.push(bcrNew);
    }

    if (dirtyRects.length === 0) {
      for (const { record } of updates) record.dirty = false;
      return;
    }

    // ── Phase 3: clip → clearRect → redraw ───────────────────────────────
    // Build a lookup so Phase 3 uses the freshly-batched bcrNew for dirty
    // annotations (paper Algorithm 1: "draw a at BCR_new(a)"), eliminating
    // any additional getBoundingClientRect calls after the Phase 1 batch.
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

    // Redraw all annotations whose BCR intersects the dirty region.
    // Dirty annotations draw at bcrNew; non-dirty annotations draw at bcrOld.
    for (const record of this._annotations.values()) {
      const bcr = bcrNewByRecord.get(record) ?? record.bcrOld;
      if (bcr && bcr.width > 0 && bcr.height > 0 && _intersectsAny(bcr, dirtyRects)) {
        drawAnnotation(ctx, record, bcr);
      }
    }
    ctx.restore();

    // ── Phase 4: commit ───────────────────────────────────────────────────
    for (const { record, bcrNew } of updates) {
      record.bcrOld  = bcrNew.width > 0 ? bcrNew : null;
      record.dirty   = false;
    }
  }

  destroy() {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }
}

/**
 * Returns true if rect a intersects any rect in the list.
 */
function _intersectsAny(a, rects) {
  for (const b of rects) {
    if (a.x < b.x + b.width  &&
        a.x + a.width  > b.x &&
        a.y < b.y + b.height &&
        a.y + a.height > b.y) {
      return true;
    }
  }
  return false;
}
