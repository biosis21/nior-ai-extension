"""
NIOR-AI Feature Extraction Pipeline (Python / Playwright)
==========================================================
Extracts the 72-dimensional feature vectors from web pages for training
the NIOR-AI classifier. Uses Playwright to render pages in a real Chromium
instance so all DOM properties reflect a fully-initialised page state.

Usage
-----
    python extract_features.py \
        --urls   data/urls.txt \
        --labels data/labels.jsonl \
        --output data/webui_augmented.csv \
        --workers 4

Label format (labels.jsonl)
---------------------------
Each line is a JSON object:
    {"url": "https://...", "selector": "#hero", "label": "call_to_action"}

Requirements
------------
    pip install playwright pandas numpy tqdm
    playwright install chromium
"""

import argparse
import asyncio
import json
import math
import pathlib
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd
from playwright.async_api import async_playwright, Page, ElementHandle
from tqdm.asyncio import tqdm_asyncio

LABELS = [
    'main_content', 'navigation', 'call_to_action', 'form',
    'heading', 'sidebar', 'media', 'advertisement'
]

KEYWORDS = [
    'nav','navigation','sidebar','side-bar','hero','cta',
    'footer','header','form','ad','advertisement','banner',
    'content','main','modal','popup','menu','dropdown',
    'carousel','slider','widget','card','article','post',
    'gallery','media','video','image','search','login',
    'signup','checkout'
]

TAG_ORDER = ['nav','header','footer','main','aside','section','article','form',
             'button','input','select','a','h','other']

ARIA_ORDER = ['navigation','main','banner','contentinfo',
              'complementary','form','search','none']

# ── Page-level helpers ────────────────────────────────────────────────────────

FEATURE_JS = """
(el) => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const scrollH = document.documentElement.scrollHeight || 1;
  const bcr = el.getBoundingClientRect();
  const tag = el.tagName.toLowerCase();
  const role = (el.getAttribute('role') || tag).toLowerCase();
  const text = el.innerText || '';
  const words = text.trim().split(/\\s+/).filter(Boolean);
  const links = el.querySelectorAll('a').length;
  const media = el.querySelectorAll('img,video,audio,canvas,svg').length;
  const interactive = el.querySelectorAll(
    'button,input,select,textarea,[role="button"],[tabindex]').length;
  const children = el.children.length;
  const descendants = el.querySelectorAll('*').length;
  let depth = 0, p = el;
  while (p.parentElement) { depth++; p = p.parentElement; }
  const tokens = new Set(
    ((el.className||'')+' '+(el.id||'')).toLowerCase().split(/[\\s\\-_]+/)
  );
  const pageTop = bcr.top + (window.scrollY || 0);
  let ancestorArea = vw * vh;
  const BLOCK = new Set(['DIV','SECTION','ARTICLE','ASIDE','MAIN',
                          'HEADER','FOOTER','NAV','FORM','LI','TD','TH']);
  let anc = el.parentElement;
  while (anc && anc !== document.body) {
    if (BLOCK.has(anc.tagName)) {
      const ar = anc.getBoundingClientRect();
      if (ar.width * ar.height > 0) { ancestorArea = ar.width * ar.height; break; }
    }
    anc = anc.parentElement;
  }
  return {
    tag, role, depth,
    bcr_x: bcr.left, bcr_y: bcr.top, bcr_w: bcr.width, bcr_h: bcr.height,
    viewport_w: vw, viewport_h: vh, scroll_h: scrollH, page_top: pageTop,
    chars: text.length, words: words.length, links,
    media, interactive, children, descendants,
    ancestor_area: ancestorArea,
    class_id: (el.className||'')+' '+(el.id||''),
    tokens: [...tokens]
  };
}
"""

def build_feature_vector(data: dict) -> np.ndarray:
    vec = np.zeros(72, dtype=np.float32)
    tag = data['tag']
    tag_key = 'h' if re.match(r'^h[1-6]$', tag) else (tag if tag in TAG_ORDER else 'other')
    ti = TAG_ORDER.index(tag_key) if tag_key in TAG_ORDER else TAG_ORDER.index('other')
    vec[ti] = 1

    role = data['role']
    ai = ARIA_ORDER.index(role) if role in ARIA_ORDER else ARIA_ORDER.index('none')
    vec[14 + ai] = 1

    # dim 21: normalised depth (use 8 as rough median fallback)
    vec[21] = min(data['depth'] / 8.0, 3.0)

    vw, vh = data['viewport_w'], data['viewport_h']
    bcr_area = data['bcr_w'] * data['bcr_h']
    vec[22] = data['bcr_x'] / vw if vw > 0 else 0
    vec[23] = data['bcr_y'] / vh if vh > 0 else 0
    vec[24] = data['bcr_w'] / vw if vw > 0 else 0
    vec[25] = data['bcr_h'] / vh if vh > 0 else 0
    vec[26] = bcr_area / (vw * vh) if vw * vh > 0 else 0
    vec[27] = data['page_top'] / data['scroll_h'] if data['scroll_h'] > 0 else 0
    vec[28] = 1.0 if 0 <= data['bcr_y'] < vh else 0.0
    vec[29] = bcr_area / data['ancestor_area'] if data['ancestor_area'] > 0 else 0

    vec[30] = math.log1p(data['chars'])
    vec[31] = math.log1p(data['words'])
    vec[32] = math.log1p(data['links'])
    vec[33] = data['links'] / data['words'] if data['words'] > 0 else 0
    vec[34] = math.log1p(data['media'])
    vec[35] = math.log1p(data['interactive'])
    vec[36] = math.log1p(data['children'])
    vec[37] = math.log1p(data['descendants'])

    tokens = set(data.get('tokens', []))
    for i, kw in enumerate(KEYWORDS):
        vec[38 + i] = 1.0 if kw.replace('-', '_').replace('-', '') in tokens else 0.0

    return vec

# ── Extraction worker ─────────────────────────────────────────────────────────

async def extract_page(page: Page, url: str,
                       page_labels: list[dict]) -> list[dict]:
    rows = []
    try:
        await page.goto(url, wait_until='networkidle', timeout=30_000)
        await page.wait_for_timeout(1_000)  # allow late JS mutations

        for item in page_labels:
            try:
                el: Optional[ElementHandle] = await page.query_selector(item['selector'])
                if el is None:
                    continue
                data = await page.evaluate(FEATURE_JS, el)
                vec = build_feature_vector(data)
                row = {col: float(vec[i]) for i, col in enumerate(col_names())}
                row['label'] = item['label']
                row['url']   = url
                rows.append(row)
            except Exception as e:
                print(f"  [warn] {url} / {item['selector']}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"  [error] {url}: {e}", file=sys.stderr)
    return rows

def col_names():
    from train import feature_columns
    return feature_columns()

async def run(args):
    urls_file  = pathlib.Path(args.urls)
    labels_file = pathlib.Path(args.labels)

    urls = [u.strip() for u in urls_file.read_text().splitlines() if u.strip()]
    label_map: dict[str, list] = {}
    for line in labels_file.read_text().splitlines():
        if not line.strip(): continue
        item = json.loads(line)
        label_map.setdefault(item['url'], []).append(item)

    all_rows = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        sem = asyncio.Semaphore(args.workers)

        async def process(url):
            async with sem:
                page = await browser.new_page(viewport={'width': 1280, 'height': 800})
                rows = await extract_page(page, url, label_map.get(url, []))
                await page.close()
                return rows

        tasks = [process(u) for u in urls]
        results = await tqdm_asyncio.gather(*tasks, desc='Extracting')
        for r in results:
            all_rows.extend(r)

        await browser.close()

    df = pd.DataFrame(all_rows)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n[✓] Saved {len(df):,} rows to {args.output}")
    print(df['label'].value_counts().to_string())

def main():
    parser = argparse.ArgumentParser(description='Extract NIOR-AI training features')
    parser.add_argument('--urls',    default='data/urls.txt')
    parser.add_argument('--labels',  default='data/labels.jsonl')
    parser.add_argument('--output',  default='data/webui_augmented.csv')
    parser.add_argument('--workers', type=int, default=4)
    asyncio.run(run(parser.parse_args()))

if __name__ == '__main__':
    main()
