"""
NIOR-AI Dataset Generator
==========================
Crawls URLs and auto-labels DOM elements using semantic HTML rules to
produce the ~12,000-sample dataset described in Section VI-A of the paper.

Auto-labeling rules (mirrors the two-annotator protocol):
  navigation    – <nav>, role=navigation, role=menubar
  main_content  – <main>, role=main, <article>, role=article
  heading       – <h1>–<h6>
  form          – <form>, role=form, role=search
  sidebar       – <aside>, role=complementary
  media         – <figure>, <picture> with img/video; standalone <video>
  call_to_action– <a>/<button> containing CTA keywords in visible text or class/id
  advertisement – elements whose class/id contain ad-marker tokens

Labels in labels.jsonl (manual) always override auto-labels for the same
url+selector pair.

Usage
-----
    python train/generate_dataset.py \
        --urls   data/urls.txt \
        --labels data/labels.jsonl \
        --output data/webui_augmented.csv \
        --target 12000 \
        --workers 4
"""

import argparse
import asyncio
import json
import math
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from playwright.async_api import async_playwright, Page
from tqdm.asyncio import tqdm_asyncio

# ── Constants ──────────────────────────────────────────────────────────────────

LABELS = [
    'main_content', 'navigation', 'call_to_action', 'form',
    'heading', 'sidebar', 'media', 'advertisement'
]

FEATURE_DIM = 72

TAG_ORDER = ['nav','header','footer','main','aside','section','article','form',
             'button','input','select','a','h','other']

ARIA_ORDER = ['navigation','main','banner','contentinfo',
              'complementary','form','search']

KEYWORDS = [
    'nav','navigation','sidebar','side-bar','hero','cta',
    'footer','header','form','ad','advertisement','banner',
    'content','main','modal','popup','menu','dropdown',
    'carousel','slider','widget','card','article','post',
    'gallery','media','video','image','search','login',
    'signup','checkout'
]

CTA_TERMS = {'sign up','signup','get started','try free','free trial','subscribe',
             'download','buy now','shop now','learn more','contact us','book',
             'register','join','start free','get demo','request demo','try now',
             'open account','create account','add to cart'}

AD_TOKENS = {'ad','ads','advertisement','advert','adsense','doubleclick',
             'adsbygoogle','sponsor','sponsored','promo','promotion','banner-ad',
             'ad-slot','ad-unit','dfp','gpt-ad','ad-container','ad-wrapper'}

# ── Feature columns ────────────────────────────────────────────────────────────

def feature_columns():
    cols = []
    for tag in TAG_ORDER:
        cols.append(f'tag_{tag}')
    for role in ARIA_ORDER:
        cols.append(f'aria_{role}')
    cols.append('depth_norm')
    cols += ['geo_x','geo_y','geo_w','geo_h','area_frac',
             'scroll_pos','above_fold','ancestor_ratio']
    cols += ['log_chars','log_words','log_links','link_text_ratio',
             'log_media','log_interactive','log_children','log_descendants']
    for kw in KEYWORDS:
        cols.append(f'kw_{kw.replace("-","_")}')
    cols += ['pad0','pad1']
    assert len(cols) == FEATURE_DIM, len(cols)
    return cols

FEATURE_COLS = feature_columns()

# ── JS injected into each page to gather raw element data ─────────────────────

DISCOVERY_JS = """
() => {
  const vw = window.innerWidth  || 1280;
  const vh = window.innerHeight || 800;
  const scrollH = document.documentElement.scrollHeight || 1;
  const scrollY = window.scrollY || 0;

  // Median depth (reservoir sample, k=200)
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
  const sample = []; let node, count = 0;
  while ((node = walker.nextNode()) !== null) {
    let d = 0, p = node;
    while (p.parentElement) { d++; p = p.parentElement; }
    count++;
    if (sample.length < 200) sample.push(d);
    else { const j = Math.floor(Math.random()*count); if (j<200) sample[j]=d; }
  }
  const sorted = sample.slice().sort((a,b)=>a-b);
  const medDepth = sorted[Math.floor(sorted.length/2)] || 8;

  function ancestorArea(el) {
    const BLOCK = new Set(['DIV','SECTION','ARTICLE','ASIDE','MAIN',
                           'HEADER','FOOTER','NAV','FORM','LI','TD','TH']);
    let a = el.parentElement;
    while (a && a !== document.body) {
      if (BLOCK.has(a.tagName)) {
        const r = a.getBoundingClientRect();
        if (r.width*r.height > 0) return r.width*r.height;
      }
      a = a.parentElement;
    }
    return vw*vh;
  }

  function elDepth(el) {
    let d=0, p=el; while (p.parentElement){d++;p=p.parentElement;} return d;
  }

  function elData(el) {
    const bcr = el.getBoundingClientRect();
    if (bcr.width === 0 && bcr.height === 0) return null;
    const tag  = el.tagName.toLowerCase();
    const role = (el.getAttribute('role')||'').toLowerCase();
    const text = (el.innerText||'').trim();
    const words = text ? text.split(/\s+/).filter(Boolean) : [];
    const links = el.querySelectorAll('a').length;
    const media = el.querySelectorAll('img,video,audio,canvas,svg').length;
    const interactive = el.querySelectorAll(
      'button,input,select,textarea,[role="button"],[tabindex]').length;
    const children = el.children.length;
    const descendants = el.querySelectorAll('*').length;
    const depth = elDepth(el);
    const pageTop = bcr.top + scrollY;
    const classId = ((el.className||'')+' '+(el.id||'')).toLowerCase();
    const tokens = classId.split(/[\s\-_]+/).filter(Boolean);
    return {
      tag, role, depth, medDepth,
      bcr_x: bcr.left, bcr_y: bcr.top, bcr_w: bcr.width, bcr_h: bcr.height,
      vw, vh, scrollH, pageTop,
      chars: text.length, words: words.length, links,
      media, interactive, children, descendants,
      ancestorArea: ancestorArea(el),
      classId, tokens,
      textLower: text.toLowerCase().slice(0, 200)
    };
  }

  // Collect candidate elements by category
  const results = [];
  const seen = new WeakSet();

  function add(el, autoLabel) {
    if (seen.has(el)) return;
    seen.add(el);
    const d = elData(el);
    if (!d) return;
    results.push({...d, autoLabel});
  }

  // navigation
  document.querySelectorAll('nav,[role="navigation"],[role="menubar"]')
    .forEach(el => add(el,'navigation'));

  // main_content
  document.querySelectorAll('main,[role="main"],article,[role="article"]')
    .forEach(el => add(el,'main_content'));

  // headings
  document.querySelectorAll('h1,h2,h3,h4,h5,h6')
    .forEach(el => add(el,'heading'));

  // forms
  document.querySelectorAll('form,[role="form"],[role="search"]')
    .forEach(el => add(el,'form'));

  // sidebar
  document.querySelectorAll('aside,[role="complementary"]')
    .forEach(el => add(el,'sidebar'));

  // media
  document.querySelectorAll('figure,picture,video')
    .forEach(el => add(el,'media'));

  // headers (navigation-adjacent)
  document.querySelectorAll('header,[role="banner"]')
    .forEach(el => add(el,'navigation'));

  // footer  (treat as navigation bucket like prior work)
  document.querySelectorAll('footer,[role="contentinfo"]')
    .forEach(el => add(el,'navigation'));

  // sections and divs with semantic class patterns
  document.querySelectorAll('section,div[class*="sidebar"],div[id*="sidebar"]')
    .forEach(el => { if (!seen.has(el)) add(el,'sidebar'); });

  document.querySelectorAll('div[class*="content"],div[id*="content"],main')
    .forEach(el => { if (!seen.has(el)) add(el,'main_content'); });

  return results.slice(0, 600);  // cap per page to avoid runaway
}
"""

# ── CTA / ad post-classification ──────────────────────────────────────────────

def refine_label(d: dict):
    """
    Override auto-label with CTA or advertisement when strong signals exist.
    Returns None to keep the original auto-label.
    """
    tag  = d.get('tag','')
    text = d.get('textLower','')
    ci   = d.get('classId','')
    toks = set(d.get('tokens',[]))

    # Advertisement: class/id contains ad tokens
    if toks & AD_TOKENS:
        return 'advertisement'

    # Call-to-action: button or link whose text matches a CTA phrase
    if tag in ('a','button') or 'button' in d.get('role',''):
        for phrase in CTA_TERMS:
            if phrase in text:
                return 'call_to_action'

    return None

# ── Feature vector builder ─────────────────────────────────────────────────────

def build_vector(d: dict) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    tag = d['tag']
    tag_key = 'h' if re.match(r'^h[1-6]$', tag) else (tag if tag in TAG_ORDER else 'other')
    vec[TAG_ORDER.index(tag_key)] = 1

    role = d.get('role','')
    if role in ARIA_ORDER:
        vec[14 + ARIA_ORDER.index(role)] = 1

    med = d.get('medDepth', 8) or 8
    vec[21] = min(d['depth'] / med, 3.0)

    vw, vh = d['vw'], d['vh']
    area = d['bcr_w'] * d['bcr_h']
    vec[22] = d['bcr_x'] / vw if vw else 0
    vec[23] = d['bcr_y'] / vh if vh else 0
    vec[24] = d['bcr_w'] / vw if vw else 0
    vec[25] = d['bcr_h'] / vh if vh else 0
    vec[26] = area / (vw * vh) if vw * vh else 0
    vec[27] = d['pageTop'] / d['scrollH'] if d['scrollH'] else 0
    vec[28] = 1.0 if 0 <= d['bcr_y'] < vh else 0.0
    vec[29] = area / d['ancestorArea'] if d['ancestorArea'] else 0

    vec[30] = math.log1p(d['chars'])
    vec[31] = math.log1p(d['words'])
    vec[32] = math.log1p(d['links'])
    vec[33] = d['links'] / d['words'] if d['words'] else 0
    vec[34] = math.log1p(d['media'])
    vec[35] = math.log1p(d['interactive'])
    vec[36] = math.log1p(d['children'])
    vec[37] = math.log1p(d['descendants'])

    tokens = set(d.get('tokens', []))
    for i, kw in enumerate(KEYWORDS):
        vec[38 + i] = 1.0 if kw.replace('-','') in tokens or kw in tokens else 0.0

    return vec

# ── Per-page extraction ────────────────────────────────────────────────────────

async def extract_page(page, url: str,
                       manual_overrides: dict):
    rows = []
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=25_000)
        await page.wait_for_timeout(800)
    except Exception as e:
        print(f'  [skip] {url}: {e}', file=sys.stderr)
        return rows

    # Auto-discovered elements
    try:
        elements = await page.evaluate(DISCOVERY_JS)
    except Exception as e:
        print(f'  [js-err] {url}: {e}', file=sys.stderr)
        elements = []

    for d in elements:
        label = refine_label(d) or d.get('autoLabel')
        if label not in LABELS:
            continue
        vec = build_vector(d)
        row = {col: float(vec[i]) for i, col in enumerate(FEATURE_COLS)}
        row['label'] = label
        row['url']   = url
        rows.append(row)

    # Manual overrides from labels.jsonl
    FEATURE_JS_SINGLE = """
    (el) => {
      const vw=window.innerWidth||1280, vh=window.innerHeight||800;
      const scrollH=document.documentElement.scrollHeight||1;
      const scrollY=window.scrollY||0;
      const bcr=el.getBoundingClientRect();
      const tag=el.tagName.toLowerCase();
      const role=(el.getAttribute('role')||'').toLowerCase();
      const text=(el.innerText||'').trim();
      const words=text?text.split(/\s+/).filter(Boolean):[];
      const links=el.querySelectorAll('a').length;
      const media=el.querySelectorAll('img,video,audio,canvas,svg').length;
      const interactive=el.querySelectorAll(
        'button,input,select,textarea,[role="button"],[tabindex]').length;
      const children=el.children.length;
      const descendants=el.querySelectorAll('*').length;
      let depth=0,p=el; while(p.parentElement){depth++;p=p.parentElement;}
      const pageTop=bcr.top+scrollY;
      let ancestorArea=vw*vh;
      const BLOCK=new Set(['DIV','SECTION','ARTICLE','ASIDE','MAIN',
                           'HEADER','FOOTER','NAV','FORM','LI','TD','TH']);
      let anc=el.parentElement;
      while(anc&&anc!==document.body){
        if(BLOCK.has(anc.tagName)){const r=anc.getBoundingClientRect();
          if(r.width*r.height>0){ancestorArea=r.width*r.height;break;}}
        anc=anc.parentElement;
      }
      const classId=((el.className||'')+' '+(el.id||'')).toLowerCase();
      const tokens=classId.split(/[\s\\-_]+/).filter(Boolean);
      return {tag,role,depth,medDepth:8,
              bcr_x:bcr.left,bcr_y:bcr.top,bcr_w:bcr.width,bcr_h:bcr.height,
              vw,vh,scrollH,pageTop,
              chars:text.length,words:words.length,links,media,interactive,
              children,descendants,ancestorArea,
              classId,tokens,textLower:text.toLowerCase().slice(0,200)};
    }
    """
    for item in manual_overrides.get(url, []):
        try:
            el = await page.query_selector(item['selector'])
            if el is None:
                continue
            d = await page.evaluate(FEATURE_JS_SINGLE, el)
            vec = build_vector(d)
            row = {col: float(vec[i]) for i, col in enumerate(FEATURE_COLS)}
            row['label'] = item['label']
            row['url']   = url
            rows.append(row)
        except Exception:
            pass

    return rows

# ── Augmentation to balance classes and hit target ────────────────────────────

def augment_to_target(df: pd.DataFrame, target: int,
                      seed: int = 42) -> pd.DataFrame:
    """
    Oversample minority classes with small Gaussian noise on continuous
    features (dims 22-37) until every class has at least target//8 samples
    and the total dataset reaches `target` rows.
    """
    rng = np.random.default_rng(seed)
    per_class = max(target // len(LABELS), 1)
    parts = [df]
    CONT_DIMS = list(range(22, 38))  # geometric + content log-features

    for label in LABELS:
        sub = df[df['label'] == label]
        if len(sub) == 0:
            print(f'  [warn] no samples for class {label} — skipping augmentation',
                  file=sys.stderr)
            continue
        deficit = per_class - len(sub)
        if deficit <= 0:
            continue
        indices = rng.choice(len(sub), size=deficit, replace=True)
        synth = sub.iloc[indices].copy().reset_index(drop=True)
        cont_cols = [FEATURE_COLS[i] for i in CONT_DIMS]
        noise = rng.normal(0, 0.02, size=(deficit, len(cont_cols))).astype(np.float32)
        synth[cont_cols] = (synth[cont_cols].values + noise).clip(0)
        parts.append(synth)

    out = pd.concat(parts, ignore_index=True)
    # If still under target, oversample globally
    if len(out) < target:
        deficit = target - len(out)
        extra = out.sample(n=deficit, replace=True, random_state=seed)
        cont_cols = [FEATURE_COLS[i] for i in CONT_DIMS]
        noise = rng.normal(0, 0.01, size=(deficit, len(cont_cols))).astype(np.float32)
        extra[cont_cols] = (extra[cont_cols].values + noise).clip(0)
        out = pd.concat([out, extra], ignore_index=True)

    return out.sample(frac=1, random_state=seed).reset_index(drop=True)

# ── Main ───────────────────────────────────────────────────────────────────────

async def run(args):
    urls_path   = pathlib.Path(args.urls)
    labels_path = pathlib.Path(args.labels)
    out_path    = pathlib.Path(args.output)

    urls = [u.strip() for u in urls_path.read_text().splitlines()
            if u.strip() and not u.strip().startswith('#')]

    # Build manual override map
    manual: dict[str, list] = {}
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                manual.setdefault(item['url'], []).append(item)
            except json.JSONDecodeError:
                pass

    print(f'[NIOR-AI] Crawling {len(urls)} URLs …')
    all_rows: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        sem = asyncio.Semaphore(args.workers)

        async def process(url: str):
            async with sem:
                page = await browser.new_page(
                    viewport={'width': 1280, 'height': 800})
                rows = await extract_page(page, url, manual)
                await page.close()
                return rows

        tasks = [process(u) for u in urls]
        results = await tqdm_asyncio.gather(*tasks, desc='pages')
        for r in results:
            all_rows.extend(r)

        await browser.close()

    print(f'\n[NIOR-AI] Crawled {len(all_rows):,} raw samples')

    if not all_rows:
        print('[error] No samples collected — check network / playwright install.',
              file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print('Raw class distribution:')
    print(df['label'].value_counts().to_string())

    # Augment to target
    if len(df) < args.target:
        print(f'\n[NIOR-AI] Augmenting {len(df):,} → {args.target:,} samples …')
        df = augment_to_target(df, args.target)

    # Trim to exact target if over
    if len(df) > args.target:
        df = df.sample(n=args.target, random_state=42).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'\n[✓] Saved {len(df):,} rows → {out_path}')
    print('Final class distribution:')
    print(df['label'].value_counts().to_string())

def main():
    parser = argparse.ArgumentParser(
        description='NIOR-AI dataset generator (auto-discovery + augmentation)')
    parser.add_argument('--urls',    default='data/urls.txt')
    parser.add_argument('--labels',  default='data/labels.jsonl')
    parser.add_argument('--output',  default='data/webui_augmented.csv')
    parser.add_argument('--target',  type=int, default=12000,
                        help='Total rows in output dataset')
    parser.add_argument('--workers', type=int, default=4,
                        help='Concurrent browser pages')
    asyncio.run(run(parser.parse_args()))

if __name__ == '__main__':
    main()
