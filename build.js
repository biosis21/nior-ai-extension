import esbuild from 'esbuild';
import { cpSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'fs';

mkdirSync('dist', { recursive: true });

// Bundle content script (no ES module support in content scripts)
await esbuild.build({
  entryPoints: ['src/content/content.js'],
  bundle: true,
  outfile: 'dist/content.js',
  format: 'iife',
  platform: 'browser',
  target: 'chrome112',
});

// Bundle offscreen script
await esbuild.build({
  entryPoints: ['src/offscreen/offscreen.js'],
  bundle: true,
  outfile: 'dist/offscreen.js',
  format: 'iife',
  platform: 'browser',
  target: 'chrome112',
});

// Service worker supports module type — copy as-is
cpSync('src/background/service-worker.js', 'dist/service-worker.js');

// Write dist/manifest.json with paths rewritten for the dist/ folder
const manifest = JSON.parse(readFileSync('manifest.json', 'utf8'));
manifest.background.service_worker = 'service-worker.js';
manifest.content_scripts[0].js = ['content.js'];
writeFileSync('dist/manifest.json', JSON.stringify(manifest, null, 2));

// Copy static assets
cpSync('popup.html', 'dist/popup.html');
// Rewrite offscreen.html: fix script src for the dist/ root
writeFileSync(
  'dist/offscreen.html',
  readFileSync('offscreen.html', 'utf8').replace('src="dist/offscreen.js"', 'src="offscreen.js"')
);
cpSync('icons', 'dist/icons', { recursive: true });
cpSync('wasm',  'dist/wasm',  { recursive: true });

// Copy onnxruntime-web wasm binaries into dist/wasm/
const ortWasmSrc = 'node_modules/onnxruntime-web/dist';
mkdirSync('dist/wasm', { recursive: true });
for (const f of readdirSync(ortWasmSrc)) {
  if (f.endsWith('.wasm') || f.endsWith('.mjs')) cpSync(`${ortWasmSrc}/${f}`, `dist/wasm/${f}`);
}

console.log('Build complete → dist/');
