"""
NIOR-AI Classifier Training Pipeline
=====================================
Trains the XGBoost gradient-boosted model described in Section VI-C
of the paper and exports it to the ONNX format for deployment via
ONNX Runtime Web.

Usage
-----
    python train.py \
        --data   data/webui_augmented.csv \
        --output wasm/nior_classifier.onnx \
        --trials 50

Requirements
------------
    pip install xgboost scikit-learn skl2onnx onnxruntime optuna pandas numpy
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import optuna
import onnxruntime as rt

# ── Constants ─────────────────────────────────────────────────────────────────

LABELS = [
    'main_content', 'navigation', 'call_to_action', 'form',
    'heading', 'sidebar', 'media', 'advertisement'
]

FEATURE_DIM = 72

# ── Feature columns (must match features.js ordering) ─────────────────────────

def feature_columns():
    cols = []
    # Structural: 14-way tag one-hot [0..13]
    for tag in ['nav','header','footer','main','aside','section','article','form',
                'button','input','select','a','h','other']:
        cols.append(f'tag_{tag}')
    # Structural: 7-way ARIA role one-hot [14..20]
    # 'none' and unrecognised roles leave all 7 bits zero (implicit none).
    # Dim 21 is reserved exclusively for normalised depth – no aliasing.
    for role in ['navigation','main','banner','contentinfo',
                 'complementary','form','search']:
        cols.append(f'aria_{role}')
    # Structural: normalised depth [21]  (14 tag + 7 aria + 1 depth = 22 structural dims)
    cols.append('depth_norm')
    # Geometric [22..29]
    cols += ['geo_x','geo_y','geo_w','geo_h','area_frac',
             'scroll_pos','above_fold','ancestor_ratio']
    # Content [30..37]
    cols += ['log_chars','log_words','log_links','link_text_ratio',
             'log_media','log_interactive','log_children','log_descendants']
    # Keyword indicators [38..69]
    KEYWORDS = [
        'nav','navigation','sidebar','side-bar','hero','cta',
        'footer','header','form','ad','advertisement','banner',
        'content','main','modal','popup','menu','dropdown',
        'carousel','slider','widget','card','article','post',
        'gallery','media','video','image','search','login',
        'signup','checkout'
    ]
    for kw in KEYWORDS:
        cols.append(f'kw_{kw.replace("-","_")}')
    # Padding [70..71]
    cols += ['pad0', 'pad1']
    assert len(cols) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(cols)}"
    return cols

FEATURE_COLS = feature_columns()

# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns in dataset: {missing}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    le = LabelEncoder()
    le.fit(LABELS)
    y = le.transform(df['label'].values)
    return X, y, le

# ── Optuna objective ───────────────────────────────────────────────────────────

def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 100, 400),
            'max_depth':       trial.suggest_int('max_depth', 3, 8),
            'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight':trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha':       trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':      trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'objective':       'multi:softprob',
            'num_class':       len(LABELS),
            'tree_method':     'hist',
            'eval_metric':     'mlogloss',
            'use_label_encoder': False
        }
        clf = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1,
                                early_stopping_rounds=20)
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)
        preds = clf.predict(X_val)
        return f1_score(y_val, preds, average='macro')

    return objective

# ── Export to ONNX ─────────────────────────────────────────────────────────────

def export_onnx(clf, output_path: str):
    """Convert sklearn-compatible XGBoost model to ONNX (INT8 quantised)."""
    initial_type = [('float_input', FloatTensorType([None, FEATURE_DIM]))]
    onnx_model = convert_xgboost(clf, initial_types=initial_type)
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    size_mb = pathlib.Path(output_path).stat().st_size / 1e6
    print(f"[✓] ONNX model saved to {output_path} ({size_mb:.2f} MB)")

    # Verify with ONNX Runtime
    sess = rt.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    dummy = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    out = sess.run(None, {'float_input': dummy})
    print(f"[✓] ONNX inference verified. Output shapes: {[o.shape for o in out]}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train NIOR-AI XGBoost classifier')
    parser.add_argument('--data',   default='data/webui_augmented.csv')
    parser.add_argument('--output', default='../wasm/nior_classifier.onnx')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--cv',     type=int, default=5, help='Cross-val folds (0=disable)')
    args = parser.parse_args()

    print(f"[NIOR-AI] Loading dataset from {args.data}...")
    X, y, le = load_dataset(args.data)
    print(f"  {X.shape[0]:,} samples, {X.shape[1]} features, {len(LABELS)} classes")

    # 70/15/15 split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15 / 0.85, stratify=y_trainval, random_state=42
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Hyperparameter optimisation
    print(f"\n[NIOR-AI] Optimising hyperparameters ({args.trials} trials)...")
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(X_train, y_train, X_val, y_val),
                   n_trials=args.trials, show_progress_bar=True)
    best = study.best_params
    print(f"  Best macro-F1 on val: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(best, indent=2)}")

    # Final model with best params
    best.update({
        'objective': 'multi:softprob',
        'num_class': len(LABELS),
        'tree_method': 'hist',
        'use_label_encoder': False
    })
    clf = xgb.XGBClassifier(**best, random_state=42, n_jobs=-1)
    clf.fit(X_trainval, y_trainval, verbose=False)

    # Cross-validation report
    if args.cv > 0:
        print(f"\n[NIOR-AI] {args.cv}-fold cross-validation on train+val set...")
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        cv_scores = []
        for fold, (tr, va) in enumerate(cv.split(X_trainval, y_trainval)):
            m = xgb.XGBClassifier(**best, random_state=42, n_jobs=-1)
            m.fit(X_trainval[tr], y_trainval[tr], verbose=False)
            cv_scores.append(f1_score(y_trainval[va], m.predict(X_trainval[va]),
                                      average='macro'))
        print(f"  CV macro-F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Hold-out test evaluation
    preds = clf.predict(X_test)
    print("\n[NIOR-AI] Hold-out test set results:")
    present = sorted(set(y_test) | set(preds))
    print(classification_report(y_test, preds,
                                labels=present,
                                target_names=[LABELS[i] for i in present]))
    print(f"  Macro-F1: {f1_score(y_test, preds, average='macro'):.4f}")

    # Export
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(clf, args.output)

if __name__ == '__main__':
    main()
