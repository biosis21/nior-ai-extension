/**
 * NIOR-AI Offscreen Document – ONNX Inference Worker
 *
 * Runs inside the Manifest V3 offscreen document, providing:
 *   - A persistent JS context (not subject to service worker 30s termination)
 *   - Full WebAssembly + URL.createObjectURL support
 *   - Extension-owned CSP with wasm-unsafe-eval
 *
 * Message protocol (from/to service-worker):
 *   IN  { type: 'CLASSIFY', id, batch: Float32Array[] }
 *   OUT { type: 'CLASSIFY_RESULT', id, predictions: LabelResult[] }
 *
 *   IN  { type: 'PING' }
 *   OUT { type: 'PONG' }
 */

'use strict';

import * as ort from 'onnxruntime-web';

const MODEL_URL   = chrome.runtime.getURL('wasm/nior_classifier.onnx');
const WASM_PREFIX = chrome.runtime.getURL('wasm/');

const LABELS = [
  'main_content', 'navigation', 'call_to_action', 'form',
  'heading', 'sidebar', 'media', 'advertisement'
];

const LOW_CONF_THRESHOLD = 0.65;

let ortSession = null;

/**
 * Initialise ONNX Runtime Web session once.
 */
async function initSession() {
  ort.env.wasm.wasmPaths = WASM_PREFIX;
  ort.env.wasm.simd      = true;
  ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency ?? 2, 4);

  const modelBuffer = await fetch(MODEL_URL).then(r => r.arrayBuffer());
  ortSession = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  });

  console.log('[NIOR-AI] ONNX session ready');
}

/**
 * Run inference on a batch of 72-dim feature vectors.
 *
 * @param {Float32Array[]} batch  - Array of per-element feature vectors
 * @returns {Promise<LabelResult[]>}
 */
async function runInference(batch) {
  if (!ortSession) throw new Error('Session not initialised');

  const n = batch.length;
  const flat = new Float32Array(n * 72);
  batch.forEach((v, i) => flat.set(v, i * 72));

  const tensor = new ort.Tensor('float32', flat, [n, 72]);
  const results = await ortSession.run({ float_input: tensor });

  // XGBoost ONNX output: 'label' (int64[n]) and 'probabilities' (float[n,8])
  const labels = results['label'].data;
  const probs  = results['probabilities'].data;

  return Array.from({ length: n }, (_, i) => {
    const classIdx  = Number(labels[i]);
    const probSlice = Array.from(probs.slice(i * 8, i * 8 + 8));
    const maxProb   = Math.max(...probSlice);
    return {
      label:      LABELS[classIdx] ?? 'main_content',
      confidence: maxProb,
      lowConf:    maxProb < LOW_CONF_THRESHOLD,
      probs:      probSlice
    };
  });
}

// ── Gemini Nano supplementary channel ────────────────────────────────────────

/**
 * Serialise a feature vector to a human-readable JSON object for
 * the Gemini Nano Prompt API. Sends only numeric/categorical metadata;
 * no raw page text is included.
 */
function featureToJSON(features, elementMeta) {
  return JSON.stringify({
    tag:          elementMeta.tag,
    aria_role:    elementMeta.ariaRole,
    area_frac:    +features[26].toFixed(4),
    link_count:   Math.round(Math.exp(features[32]) - 1),
    interactive:  Math.round(Math.exp(features[35]) - 1),
    children:     Math.round(Math.exp(features[36]) - 1),
    above_fold:   features[28],
    depth_norm:   +features[21].toFixed(3)
  });
}

const GEMINI_PROMPT_TMPL = (json) =>
  `Given the following DOM element properties, classify the element into ` +
  `exactly one of [main_content, navigation, call_to_action, form, heading, ` +
  `sidebar, media, advertisement]. Respond only with a JSON object ` +
  `{"category": string, "confidence": number}.\n\nElement: ${json}`;

/**
 * Query Gemini Nano for a single low-confidence element.
 * Returns null if the API is unavailable.
 */
async function queryGeminiNano(features, elementMeta) {
  if (typeof window.ai?.languageModel === 'undefined') return null;

  try {
    const session = await window.ai.languageModel.create({
      systemPrompt: 'You are a precise DOM element classifier. Always respond with valid JSON only.'
    });
    const response = await session.prompt(GEMINI_PROMPT_TMPL(
      featureToJSON(features, elementMeta)
    ), {
      responseConstraint: {
        type: 'object',
        properties: {
          category:   { type: 'string', enum: LABELS },
          confidence: { type: 'number' }
        },
        required: ['category', 'confidence']
      }
    });
    session.destroy();
    const parsed = JSON.parse(response);
    return { label: parsed.category, confidence: parsed.confidence };
  } catch {
    return null;
  }
}

// ── Message handler ───────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === 'PING') {
    sendResponse({ type: 'PONG' });
    return false;
  }

  if (msg.type === 'CLASSIFY') {
    const { id, batch, metas } = msg;
    const features = batch.map(b => new Float32Array(b));

    (async () => {
      try {
        let predictions = await runInference(features);

        // Tier-2: Gemini Nano for low-confidence elements
        const lowConfIdxs = predictions
          .map((p, i) => p.lowConf ? i : -1)
          .filter(i => i !== -1);

        if (lowConfIdxs.length > 0) {
          await Promise.all(lowConfIdxs.map(async (i) => {
            const geminiResult = await queryGeminiNano(features[i], metas[i]);
            if (geminiResult) {
              predictions[i] = { ...predictions[i], ...geminiResult, tier: 2 };
            }
          }));
        }

        sendResponse({ type: 'CLASSIFY_RESULT', id, predictions });
      } catch (err) {
        console.error('[NIOR-AI] Inference error:', err);
        sendResponse({ type: 'CLASSIFY_RESULT', id, predictions: [] });
      }
    })();

    return true; // keep message channel open for async response
  }
});

// Initialise immediately when the offscreen document is created
initSession().catch(err => console.error('[NIOR-AI] Init failed:', err));
