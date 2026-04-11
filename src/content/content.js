/**
 * NIOR-AI Content Script
 *
 * Entry point injected into every page frame. Coordinates the three
 * subsystems:
 *   1. CandidateCollector  – gate-filtered element discovery
 *   2. Visibility filter   – 5-phase composite visibility check
 *   3. Feature extractor   – 72-dim DOM feature extraction
 *
 * Feature batches are forwarded to the service worker, which routes them
 * to the offscreen document for ONNX inference. Classification results
 * are returned and passed to NIOR_AI.batch() for shadow-canvas rendering.
 *
 * The content script never writes to the host page's light DOM.
 */

'use strict';

import { CandidateCollector, syncVisibilityFilter } from '../classifier/visibility.js';
import { batchExtract } from '../classifier/features.js';
import { ShadowRenderer } from '../shadow-renderer/renderer.js';

let renderer  = null;
let collector = null;

// Map: element → annotation handle (for update/remove on re-classification)
const elementHandles = new Map();

// ── Initialisation ────────────────────────────────────────────────────────────

function init() {
  renderer = new ShadowRenderer();

  collector = new CandidateCollector({
    onBatch: handleCandidateBatch,
    nMin: 5,
    debounceMs: 300,
    batchSize: 20
  });

  // Listen for classification results from the service worker
  chrome.runtime.onMessage.addListener(handleMessage);
}

// ── Candidate processing ──────────────────────────────────────────────────────

async function handleCandidateBatch(elements) {
  // Run phases 1, 2, 4, 5 (phase 3 handled by CandidateCollector's IO)
  const visible = await syncVisibilityFilter(
    elements.map(el => ({ element: el }))
  );
  if (visible.length === 0) return;

  // Extract features (batch read, no writes)
  const featureData = batchExtract(visible);

  // Build minimal element metadata for Gemini Nano serialisation
  const metas = featureData.map(({ element }) => ({
    tag:      element.tagName.toLowerCase(),
    ariaRole: element.getAttribute('role') || ''
  }));

  // Forward to service worker → offscreen document
  chrome.runtime.sendMessage({
    type:  'CLASSIFY_REQUEST',
    batch: featureData.map(({ features }) => Array.from(features)),
    metas,
    // Include element indices so we can correlate results
    elementIds: featureData.map((_, i) => i)
  }, (response) => {
    if (chrome.runtime.lastError || !response) return;
    applyPredictions(featureData.map(d => d.element), response.predictions);
  });
}

// ── Render results ────────────────────────────────────────────────────────────

function applyPredictions(elements, predictions) {
  renderer.batch(() => {
    elements.forEach((el, i) => {
      const { label, confidence } = predictions[i];

      // Remove existing annotation for this element if any
      const existing = elementHandles.get(el);
      if (existing !== undefined) renderer.remove(existing);

      const handle = renderer.annotate(el, {
        type:     'bbox',
        label,
        color:    labelColor(label),
        labelPos: 'top-left',
        fill:     confidence > 0.8
      });
      elementHandles.set(el, handle);
    });
  });
}

// ── Message handler ───────────────────────────────────────────────────────────

function handleMessage(msg) {
  if (msg.type === 'NIOR_CLEAR') {
    renderer?.clear();
    elementHandles.clear();
  }
  if (msg.type === 'NIOR_TOGGLE') {
    toggleOverlay(msg.visible);
  }
}

function toggleOverlay(visible) {
  // Implemented via canvas display property inside shadow root
  // (safe: touches only shadow-internal elements)
  const canvas = renderer?._canvas;
  if (canvas) canvas.style.display = visible ? '' : 'none';
}

// ── Helpers ───────────────────────────────────────────────────────────────────

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

function labelColor(label) {
  return LABEL_COLORS[label] ?? '#6B7280';
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
