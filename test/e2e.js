/**
 * NIOR-AI End-to-End Playwright Test
 *
 * Strategy:
 *   - Do NOT open offscreen.html directly (Chrome allows only one offscreen
 *     document per extension — opening it via Playwright blocks the service
 *     worker from creating its own via chrome.offscreen.createDocument).
 *   - Instead: wait for the content script pipeline to run, then verify via
 *     (a) service worker evaluate and (b) screenshot pixel analysis.
 */

import { chromium } from 'playwright';
import path from 'path';
import { fileURLToPath } from 'url';
import { writeFileSync } from 'fs';
import { PNG } from 'pngjs';   // install if missing: npm i -D pngjs

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EXT_PATH  = path.resolve(__dirname, '..', 'dist');
const CLASSIFY_WAIT = 15_000;   // ONNX init can take several seconds

// Known annotation colours (hex → rgb)
const ANNOTATION_COLORS = [
  [37,  99,  235],  // main_content  #2563EB
  [22,  163,  74],  // navigation    #16A34A
  [220,  38,  38],  // call_to_action #DC2626
  [147,  51, 234],  // form          #9333EA
  [8,   145, 178],  // heading       #0891B2
  [217, 119,   6],  // sidebar       #D97706
  [219,  39, 119],  // media         #DB2777
  [100, 116, 139],  // advertisement #64748B
];
const COLOR_TOLERANCE = 15;

function hasAnnotationPixels(pngBuf) {
  const img = PNG.sync.read(pngBuf);
  for (let i = 0; i < img.data.length; i += 4) {
    const r = img.data[i], g = img.data[i+1], b = img.data[i+2];
    for (const [ar, ag, ab] of ANNOTATION_COLORS) {
      if (Math.abs(r-ar) < COLOR_TOLERANCE &&
          Math.abs(g-ag) < COLOR_TOLERANCE &&
          Math.abs(b-ab) < COLOR_TOLERANCE) return true;
    }
  }
  return false;
}

const logs = [];
function record(source, type, text) {
  logs.push({ source, type, text });
  const tag = (type === 'error' || type === 'pageerror') ? '[ERR]' : '[LOG]';
  console.log(`  [${source}]${tag} ${text}`);
}
function attach(p, label) {
  p.on('console',       msg => record(label, msg.type(), msg.text()));
  p.on('pageerror',     err => record(label, 'pageerror', err.message));
  p.on('requestfailed', req => record(label, 'error', `fail: ${req.url()}`));
  p.on('response',      r   => { if (!r.ok()) record(label, 'error', `${r.status()} ${r.url()}`); });
}

async function run() {
  console.log(`Extension: ${EXT_PATH}`);

  const context = await chromium.launchPersistentContext('', {
    headless: false,
    args: [
      `--disable-extensions-except=${EXT_PATH}`,
      `--load-extension=${EXT_PATH}`,
      '--no-sandbox',
    ],
    timeout: 30_000,
  });

  const sw = await context.waitForEvent('serviceworker');
  const extId = new URL(sw.url()).hostname;
  console.log(`Extension ID: ${extId}`);

  // Watch for any new pages (e.g., the offscreen document)
  context.on('page', p => {
    const label = p.url().includes(extId) ? `ext:${p.url().split('/').pop()}` : 'page';
    console.log(`  Page opened: ${p.url()}`);
    attach(p, label);
  });

  // Navigate to a real page — triggers content script → CLASSIFY_REQUEST pipeline
  const page = await context.newPage();
  attach(page, 'page');

  // Intercept fetch to capture 404 URLs (content script runs in page context)
  await page.addInitScript(() => {
    window.__failedUrls = [];
    const orig = window.fetch;
    window.fetch = function(url, ...a) {
      return orig.call(this, url, ...a).then(r => {
        if (!r.ok) window.__failedUrls.push(`${r.status} ${r.url}`);
        return r;
      });
    };
  });

  console.log('Navigating to example.com…');
  await page.goto('https://example.com', { waitUntil: 'domcontentloaded' });

  console.log(`Waiting up to ${CLASSIFY_WAIT / 1000}s for pipeline to complete…`);
  await page.waitForTimeout(CLASSIFY_WAIT);

  // Diagnostic: log element BCRs and canvas state
  const diagnostics = await page.evaluate(() => {
    const vw = window.innerWidth, vh = window.innerHeight, dpr = window.devicePixelRatio;
    const els = ['html','body','div','h1','p','a'];
    const bcrs = Object.fromEntries(
      els.map(tag => {
        const el = document.querySelector(tag);
        if (!el) return [tag, null];
        const r = el.getBoundingClientRect();
        return [tag, { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }];
      })
    );
    return { vw, vh, dpr, bcrs };
  });
  console.log('  Viewport:', diagnostics.vw, 'x', diagnostics.vh, 'DPR:', diagnostics.dpr);
  console.log('  Element BCRs:', JSON.stringify(diagnostics.bcrs, null, 2));

  // Capture failed fetch URLs
  const failedFetches = await page.evaluate(() => window.__failedUrls || []);
  if (failedFetches.length) {
    console.log('  Failed fetches:', failedFetches);
  }

  // Screenshot + pixel analysis
  const screenshotBuf = await page.screenshot({ type: 'png' });
  writeFileSync(path.join(__dirname, 'e2e-screenshot.png'), screenshotBuf);

  let pixelsFound = false;
  try {
    pixelsFound = hasAnnotationPixels(screenshotBuf);
  } catch {
    console.log('  (pngjs not available — skipping pixel check)');
  }

  // Results
  const initFail = logs.some(l => l.text.includes('Init failed'));
  const inferErr = logs.some(l => l.text.includes('Inference error'));
  const onnxReady= logs.some(l => l.text.includes('ONNX session ready'));
  const errors   = logs.filter(l => (l.type === 'error' || l.type === 'pageerror')
                                 && !l.text.includes('404'));
  // Filter out page-level 404s (e.g., missing favicon on the test page)
  const notFound = logs.filter(l => l.text.includes('404') && l.source !== 'page');

  console.log('\n── Results ───────────────────────────────────────');
  console.log(`ONNX ready (log)   : ${onnxReady   ? '✓' : '✗ (offscreen console not captured — normal)'}`);
  console.log(`Init failed        : ${initFail    ? '✗' : '✓'}`);
  console.log(`Inference error    : ${inferErr    ? '✗' : '✓'}`);
  console.log(`Annotation pixels  : ${pixelsFound ? '✓' : '✗'}`);
  console.log(`404s               : ${notFound.length ? notFound.map(e=>`[${e.source}] ${e.text}`).join('\n  ') : '✓ none'}`);
  console.log(`Other errors       : ${errors.length   ? errors.map(e=>`[${e.source}] ${e.text}`).join('\n  ')   : '✓ none'}`);
  console.log('Screenshot         : test/e2e-screenshot.png');
  console.log('──────────────────────────────────────────────────');

  await context.close();
  const pass = !initFail && !inferErr && pixelsFound && notFound.length === 0;
  process.exit(pass ? 0 : 1);
}

run().catch(err => { console.error(err); process.exit(1); });
