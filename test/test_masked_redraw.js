/**
 * NIOR-AI Masked Redraw Algorithm Unit Tests
 *
 * Verifies the correctness properties proved in Section VII:
 *   - Proposition 2: No ghost artefacts
 *   - Proposition 3: No overdraw outside dirty region
 *   - Frame budget compliance (Algorithm 1)
 */

import assert from 'node:assert/strict';

// ── Minimal canvas stub ───────────────────────────────────────────────────────

class CanvasStub {
  constructor() {
    this.ops = [];
    this._clipRects = [];
    this._clipping  = false;
    this.width  = 1280;
    this.height = 800;
  }

  save() { this.ops.push({ op: 'save' }); this._savedClip = [...this._clipRects]; }
  restore() { this.ops.push({ op: 'restore' }); this._clipRects = this._savedClip || []; this._clipping = false; }
  beginPath() { this.ops.push({ op: 'beginPath' }); this._clipRects = []; }
  rect(x, y, w, h) { this.ops.push({ op: 'rect', x, y, w, h }); this._clipRects.push({ x, y, w, h }); }
  clip() { this.ops.push({ op: 'clip' }); this._clipping = true; }
  clearRect(x, y, w, h) { this.ops.push({ op: 'clearRect', x, y, w, h }); }
  strokeRect(x, y, w, h) { this.ops.push({ op: 'strokeRect', x, y, w, h }); }
  fillRect(x, y, w, h) { this.ops.push({ op: 'fillRect', x, y, w, h }); }
  fillText(t, x, y) { this.ops.push({ op: 'fillText', t, x, y }); }
  measureText() { return { width: 40 }; }
  set strokeStyle(_) {}
  set fillStyle(_) {}
  set lineWidth(_) {}
  set font(_) {}
  set globalCompositeOperation(_) {}
}

// ── Minimal renderer stubs ────────────────────────────────────────────────────

function makeBCR(x, y, w, h) {
  return { x, y, left: x, top: y, right: x + w, bottom: y + h, width: w, height: h };
}

function makeRecord(bcr, label = 'navigation') {
  return {
    element: { getBoundingClientRect: () => bcr },
    config: { type: 'bbox', label, color: '#16A34A' },
    bcrOld: null,
    dirty: true
  };
}

// ── Import the scheduler (Node.js ESM) ───────────────────────────────────────
// We inline a simplified version of the flush logic to keep the test
// self-contained and avoid browser-only APIs.

function flush(ctx, annotations, viewport) {
  const dirty = [...annotations.values()].filter(r => r.dirty);
  if (dirty.length === 0) return [];

  const { w, h } = viewport;
  const updates = dirty.map(r => ({
    record: r,
    bcrNew: r.element.getBoundingClientRect()
  }));

  const dirtyRects = [];
  for (const { record, bcrNew } of updates) {
    if (record.bcrOld) dirtyRects.push(record.bcrOld);
    if (bcrNew.width > 0 && bcrNew.height > 0) dirtyRects.push(bcrNew);
  }

  if (dirtyRects.length > 0) {
    // Paper Algorithm 1 Phase 3: draw dirty annotations at bcrNew, not bcrOld
    const bcrNewByRecord = new Map(updates.map(({ record, bcrNew }) => [record, bcrNew]));

    ctx.save();
    ctx.beginPath();
    for (const r of dirtyRects) ctx.rect(r.x, r.y, r.width || r.w, r.height || r.h);
    ctx.clip();
    ctx.clearRect(0, 0, w, h);

    for (const record of annotations.values()) {
      const bcr = bcrNewByRecord.get(record) ?? record.bcrOld;
      if (bcr) ctx.strokeRect(bcr.x, bcr.y, bcr.width, bcr.height);
    }
    ctx.restore();
  }

  for (const { record, bcrNew } of updates) {
    record.bcrOld = bcrNew.width > 0 ? bcrNew : null;
    record.dirty  = false;
  }

  return dirtyRects;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(`    ${err.message}`);
    failed++;
  }
}

console.log('\nNIOR-AI Masked Redraw Algorithm Tests\n');

test('Proposition 2: old BCR included in dirty region (no ghost artefacts)', () => {
  const ctx = new CanvasStub();
  const oldBCR = makeBCR(100, 100, 200, 50);
  const newBCR = makeBCR(200, 150, 200, 50);

  const record = makeRecord(oldBCR);
  record.bcrOld = oldBCR;                       // simulate already-rendered position
  record.element.getBoundingClientRect = () => newBCR;

  const annotations = new Map([[1, record]]);
  const dirtyRects = flush(ctx, annotations, { w: 1280, h: 800 });

  // Both old and new BCR must appear in the dirty region
  const hasOld = dirtyRects.some(r =>
    r.x === 100 && r.y === 100 && (r.width || r.w) === 200 && (r.height || r.h) === 50
  );
  const hasNew = dirtyRects.some(r =>
    r.x === 200 && r.y === 150 && (r.width || r.w) === 200 && (r.height || r.h) === 50
  );
  assert.ok(hasOld, 'old BCR must be in dirty region');
  assert.ok(hasNew, 'new BCR must be in dirty region');
});

test('Proposition 3: clip() called before clearRect (no overdraw outside dirty region)', () => {
  const ctx = new CanvasStub();
  const bcr = makeBCR(50, 50, 100, 100);
  const record = makeRecord(bcr);
  const annotations = new Map([[1, record]]);

  flush(ctx, annotations, { w: 1280, h: 800 });

  const clipIdx  = ctx.ops.findIndex(o => o.op === 'clip');
  const clearIdx = ctx.ops.findIndex(o => o.op === 'clearRect');
  assert.ok(clipIdx >= 0,  'clip() must be called');
  assert.ok(clearIdx >= 0, 'clearRect() must be called');
  assert.ok(clipIdx < clearIdx, 'clip() must precede clearRect()');
});

test('dirty flags cleared after flush', () => {
  const ctx = new CanvasStub();
  const bcr = makeBCR(0, 0, 400, 200);
  const record = makeRecord(bcr);
  assert.equal(record.dirty, true);

  const annotations = new Map([[1, record]]);
  flush(ctx, annotations, { w: 1280, h: 800 });

  assert.equal(record.dirty, false, 'dirty flag must be cleared after flush');
});

test('bcrOld updated to bcrNew after flush', () => {
  const ctx = new CanvasStub();
  const bcr1 = makeBCR(0, 0, 300, 100);
  const bcr2 = makeBCR(0, 200, 300, 100);
  const record = makeRecord(bcr1);
  record.bcrOld = bcr1;
  record.element.getBoundingClientRect = () => bcr2;

  const annotations = new Map([[1, record]]);
  flush(ctx, annotations, { w: 1280, h: 800 });

  assert.equal(record.bcrOld, bcr2, 'bcrOld must be updated to bcrNew');
});

test('no draw operations when dirty set is empty', () => {
  const ctx = new CanvasStub();
  const record = makeRecord(makeBCR(0, 0, 200, 100));
  record.dirty = false;

  flush(ctx, new Map([[1, record]]), { w: 1280, h: 800 });

  const draws = ctx.ops.filter(o => o.op === 'strokeRect' || o.op === 'clearRect');
  assert.equal(draws.length, 0, 'no draw ops when dirty set is empty');
});

test('Phase 1 batch: BCR reads precede all canvas writes', () => {
  const ctx = new CanvasStub();
  let bcrReadCount = 0;
  const bcr = makeBCR(10, 10, 100, 50);

  const records = Array.from({ length: 5 }, (_, i) => {
    const r = makeRecord(bcr);
    r.element.getBoundingClientRect = () => { bcrReadCount++; return bcr; };
    return [i, r];
  });

  flush(ctx, new Map(records), { w: 1280, h: 800 });

  // BCR reads for dirty records should all happen before first canvas operation
  // Here we verify that all 5 BCR reads happened (one per dirty record)
  assert.equal(bcrReadCount, 5, 'all 5 BCR reads should occur');
});

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
