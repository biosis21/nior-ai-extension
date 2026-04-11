/**
 * NIOR-AI Feature Extractor Unit Tests
 *
 * Run with: node --experimental-vm-modules test/test_features.js
 * Or with Jest: jest test/test_features.js
 */

import { extractFeatures, batchExtract } from '../src/classifier/features.js';
import assert from 'node:assert/strict';

// ── Minimal DOM environment stubs ────────────────────────────────────────────
// (In a real test environment, use jsdom or Playwright for full fidelity)

function makeMockElement({
  tag = 'div', role = null, id = '', className = '',
  innerText = '', children = 0, depth = 3
} = {}) {
  const el = {
    tagName: tag.toUpperCase(),
    getAttribute: (attr) => attr === 'role' ? role : null,
    id,
    className,
    innerText,
    children: { length: children },
    querySelectorAll: (sel) => {
      if (sel.includes('a')) return { length: 2 };
      if (sel.includes('img')) return { length: 0 };
      if (sel.includes('button')) return { length: 1 };
      return { length: 0 };
    },
    parentElement: null
  };
  // Simulate depth by chaining parentElement
  let p = el;
  for (let i = 0; i < depth; i++) {
    const parent = { tagName: 'DIV', parentElement: null };
    p.parentElement = parent;
    p = parent;
  }
  return el;
}

function makeMockBCR(x = 0, y = 100, w = 800, h = 200) {
  return { left: x, top: y, right: x + w, bottom: y + h, width: w, height: h };
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

console.log('\nNIOR-AI Feature Extractor Tests\n');

// ─────────────────────────────────────────────────────────────────────────────

test('output is Float32Array of length 72', () => {
  const el  = makeMockElement({ tag: 'nav' });
  const bcr = makeMockBCR();
  const vec = extractFeatures(el, bcr, 1280, 800, 6000, 8);
  assert.ok(vec instanceof Float32Array, 'should be Float32Array');
  assert.equal(vec.length, 72);
});

test('tag one-hot: nav → dim 0 = 1', () => {
  const el  = makeMockElement({ tag: 'nav' });
  const vec = extractFeatures(el, makeMockBCR(), 1280, 800, 6000, 8);
  assert.equal(vec[0], 1, 'nav tag dim should be 1');
  // All other tag dims should be 0
  for (let i = 1; i < 14; i++) assert.equal(vec[i], 0, `dim ${i} should be 0`);
});

test('tag one-hot: h1 collapses to h [dim 12]', () => {
  const el = makeMockElement({ tag: 'h1' });
  const vec = extractFeatures(el, makeMockBCR(), 1280, 800, 6000, 8);
  assert.equal(vec[12], 1, 'h1 should collapse to h at dim 12');
});

test('tag one-hot: unknown tag → other [dim 13]', () => {
  const el = makeMockElement({ tag: 'figure' });
  const vec = extractFeatures(el, makeMockBCR(), 1280, 800, 6000, 8);
  assert.equal(vec[13], 1, 'unknown tag should map to other at dim 13');
});

test('ARIA role one-hot: navigation → dim 14 = 1', () => {
  const el = makeMockElement({ tag: 'div', role: 'navigation' });
  const vec = extractFeatures(el, makeMockBCR(), 1280, 800, 6000, 8);
  assert.equal(vec[14], 1, 'navigation ARIA role dim should be 1');
});

test('geometric: area_frac for full-viewport element = 1.0', () => {
  const el  = makeMockElement();
  const bcr = makeMockBCR(0, 0, 1280, 800);
  const vec = extractFeatures(el, bcr, 1280, 800, 6000, 8);
  assert.ok(Math.abs(vec[26] - 1.0) < 1e-4, `area_frac should be ~1, got ${vec[26]}`);
});

test('geometric: above_fold flag [dim 28]', () => {
  const el1 = makeMockElement();
  const bcr1 = makeMockBCR(0, 0, 400, 200); // top = 0 < 800
  const vec1 = extractFeatures(el1, bcr1, 1280, 800, 6000, 8);
  assert.equal(vec1[28], 1, 'element at top=0 should be above fold');

  const bcr2 = makeMockBCR(0, 900, 400, 200); // top = 900 > 800
  const vec2 = extractFeatures(el1, bcr2, 1280, 800, 6000, 8);
  assert.equal(vec2[28], 0, 'element at top=900 should NOT be above fold');
});

test('content: log_chars > 0 for non-empty text', () => {
  const el = makeMockElement({ innerText: 'Hello world' });
  const vec = extractFeatures(el, makeMockBCR(), 1280, 800, 6000, 8);
  assert.ok(vec[30] > 0, 'log_chars should be positive for non-empty text');
});

test('keyword indicator: "nav" in className → dim 38 = 1', () => {
  const el = makeMockElement({ className: 'main-nav sticky' });
  const vec = extractFeatures(el, makeMockBCR(), 1280, 800, 6000, 8);
  assert.equal(vec[38], 1, 'nav keyword indicator [38] should be 1');
});

test('keyword indicator: "hero" in id → dim 42 = 1', () => {
  // KEYWORDS order: nav(0), navigation(1), sidebar(2), side-bar(3), hero(4)
  // → dim 38 + 4 = 42
  const el = makeMockElement({ id: 'hero-section' });
  const vec = extractFeatures(el, makeMockBCR(), 1280, 800, 6000, 8);
  assert.equal(vec[42], 1, 'hero keyword indicator [42] should be 1');
});

test('no feature is NaN', () => {
  const el  = makeMockElement({ tag: 'article', innerText: 'Some content here.' });
  const bcr = makeMockBCR(100, 200, 600, 400);
  const vec = extractFeatures(el, bcr, 1280, 800, 6000, 8);
  for (let i = 0; i < 72; i++) {
    assert.ok(!isNaN(vec[i]), `dim ${i} is NaN`);
  }
});

test('all features in [0, ∞) for typical element', () => {
  const el  = makeMockElement({ tag: 'section', className: 'content-area', innerText: 'Text' });
  const bcr = makeMockBCR(0, 0, 800, 600);
  const vec = extractFeatures(el, bcr, 1280, 800, 6000, 8);
  for (let i = 0; i < 72; i++) {
    assert.ok(vec[i] >= 0, `dim ${i} is negative: ${vec[i]}`);
  }
});

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
