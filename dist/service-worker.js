/**
 * NIOR-AI Service Worker (Manifest V3 Background)
 *
 * Responsibilities:
 *   1. Create and keep alive the offscreen document (Section VIII-C)
 *   2. Route classification requests from content scripts → offscreen document
 *   3. Return results to the originating content script tab
 *   4. Handle extension lifecycle (install, toggle commands)
 *
 * Heartbeat: a message is sent to the offscreen document every 25 seconds
 * to prevent the service worker from being terminated by the browser's
 * 30-second idle timeout. The offscreen document responds with PONG to
 * confirm it is alive; if no response is received the document is
 * recreated on the next request.
 */

'use strict';

const OFFSCREEN_URL    = chrome.runtime.getURL('offscreen.html');
const HEARTBEAT_MS     = 25_000;
const OFFSCREEN_REASON = 'BLOBS'; // closest available reason for ONNX runtime

let offscreenReady  = false;
let heartbeatTimer  = null;

// ── Offscreen document lifecycle ─────────────────────────────────────────────

async function ensureOffscreenDocument() {
  const existing = await chrome.offscreen.hasDocument?.()
    ?? (await getExistingOffscreenDocument()) !== null;

  if (!existing) {
    await chrome.offscreen.createDocument({
      url:    OFFSCREEN_URL,
      reasons: [chrome.offscreen.Reason.BLOBS],
      justification: 'ONNX Runtime WebAssembly inference for NIOR-AI classifier'
    });
    offscreenReady = true;
    startHeartbeat();
  }
}

async function getExistingOffscreenDocument() {
  const clients = await self.clients?.matchAll({ type: 'window' });
  return clients?.find(c => c.url === OFFSCREEN_URL) ?? null;
}

function startHeartbeat() {
  clearInterval(heartbeatTimer);
  heartbeatTimer = setInterval(async () => {
    try {
      const resp = await chrome.runtime.sendMessage({ type: 'PING' });
      if (!resp || resp.type !== 'PONG') await recreateOffscreen();
    } catch {
      await recreateOffscreen();
    }
  }, HEARTBEAT_MS);
}

async function recreateOffscreen() {
  offscreenReady = false;
  try { await chrome.offscreen.closeDocument(); } catch {}
  await ensureOffscreenDocument();
}

// ── Message routing ───────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'CLASSIFY_REQUEST') {
    handleClassifyRequest(msg, sender, sendResponse);
    return true; // async response
  }
});

async function handleClassifyRequest(msg, sender, sendResponse) {
  await ensureOffscreenDocument();

  // Forward to offscreen document; relay response back to content script
  try {
    const result = await chrome.runtime.sendMessage({
      type:  'CLASSIFY',
      id:    msg.id,
      batch: msg.batch,
      metas: msg.metas
    });
    sendResponse(result);
  } catch (err) {
    console.error('[NIOR-AI SW] Classification error:', err);
    sendResponse({ type: 'CLASSIFY_RESULT', predictions: [] });
  }
}

// ── Extension install / update ────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(async () => {
  await ensureOffscreenDocument();
});

// ── Action (toolbar button) ───────────────────────────────────────────────────

// Per-tab overlay visibility state. Default: visible (true).
const _overlayVisible = new Map();

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) return;
  const current = _overlayVisible.get(tab.id) ?? true;
  const next    = !current;
  _overlayVisible.set(tab.id, next);

  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func:   (visible) => chrome.runtime.sendMessage({ type: 'NIOR_TOGGLE', visible }),
    args:   [next]
  });
});
