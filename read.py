# read.py
import argparse
from pathlib import Path
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
import joblib

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tabulate import tabulate  # <-- for pretty tables


def find_class_names(out_dir: Path):
    f = out_dir / "class_names.txt"
    if f.exists():
        return [s.strip() for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]

    m = out_dir / "train_manifest.csv"
    if m.exists():
        df = pd.read_csv(m)
        if 'label' in df.columns:
            return sorted(df['label'].unique().tolist())

    s = out_dir / "models_summary_metrics.csv"
    if s.exists():
        p = out_dir / "probs_inception_test.npy"
        if p.exists():
            probs = np.load(p)
            n = probs.shape[1]
            return [f"class_{i}" for i in range(n)]
    return None


def load_models_if_exist(out_dir: Path):
    models = {}
    scaler_path = out_dir / "scaler.joblib"
    if scaler_path.exists():
        models['scaler'] = joblib.load(scaler_path)

    for name in ("random_forest", "svm", "decision_tree"):
        p = out_dir / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)

    for name in ("random_forest_calibrated", "svm_calibrated"):
        p = out_dir / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)

    head_path = out_dir / "inception_head_model.h5"
    if head_path.exists():
        try:
            models['head'] = tf.keras.models.load_model(str(head_path))
        except Exception as e:
            print("⚠️ Could not load Keras head:", e)

    meta_path = out_dir / "meta_logistic.joblib"
    if meta_path.exists():
        models['meta'] = joblib.load(meta_path)
    return models


def extract_feature_from_image(img_path: Path, extractor):
    img = image.load_img(str(img_path), target_size=(299, 299))
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    feat = extractor.predict(arr, verbose=0)
    return feat


def print_summary(out_dir: Path):
    print("\n==================== OUTPUT SUMMARY ====================")
    print(f"📁 Contents of: {out_dir}\n")
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)

    summary = out_dir / "models_summary_metrics.csv"
    if summary.exists():
        print("\n📊 Models Summary Metrics:")
        df = pd.read_csv(summary, index_col=0)
        print(tabulate(df, headers="keys", tablefmt="pretty"))
    else:
        print("\n(no models_summary_metrics.csv found)")


def format_probs(probs, class_names):
    """Return a pretty table of probabilities sorted high → low."""
    rows = []
    for i in np.argsort(-probs):
        cname = class_names[i] if class_names else str(i)
        rows.append([cname, f"{probs[i]:.4f}"])
    return tabulate(rows, headers=["Class", "Probability"], tablefmt="pretty")


def infer_single_image(img_path: Path, out_dir: Path, class_names, models):
    extractor = InceptionV3(include_top=False, weights='imagenet',
                            pooling='avg', input_shape=(299, 299, 3))
    feat = extract_feature_from_image(img_path, extractor)

    print("\n==================== INFERENCE ====================")
    print(f"🖼️ Image: {img_path}\n")

    # --- Inception Head ---
    if 'head' in models:
        p_incep = models['head'].predict(feat, verbose=0)[0]
        idx = int(np.argmax(p_incep))
        print("🔹 Inception Head Prediction:", class_names[idx], f"(conf {p_incep[idx]:.4f})")
        print(format_probs(p_incep, class_names))
    else:
        print("❌ No Inception head model found.")

    # Scale features
    if 'scaler' in models:
        try:
            X_feat_scaled = models['scaler'].transform(feat)
        except Exception:
            X_feat_scaled = None
    else:
        X_feat_scaled = None

    # --- Classical Models ---
    for name in ('random_forest_calibrated', 'random_forest',
                 'svm_calibrated', 'svm', 'decision_tree'):
        m = models.get(name)
        if m is None:
            continue
        print(f"\n🔹 {name.upper()} Prediction:")
        try:
            if X_feat_scaled is not None and hasattr(m, "predict_proba"):
                probs = m.predict_proba(X_feat_scaled)[0]
                idx = int(np.argmax(probs))
                print(" →", class_names[idx], f"(conf {probs[idx]:.4f})")
                print(format_probs(probs, class_names))
            else:
                pred = m.predict(X_feat_scaled if X_feat_scaled is not None else feat)[0]
                print(" →", class_names[pred])
        except Exception as e:
            print(" ⚠️ Error:", e)

    # --- Meta (Stacking) ---
    if 'meta' in models:
        print("\n🔹 Meta (Stacking) Prediction:")
        try:
            p_incep = models['head'].predict(feat, verbose=0)[0] if 'head' in models else None
            p_rf = models.get('random_forest_calibrated') or models.get('random_forest')
            p_svm = models.get('svm_calibrated') or models.get('svm')

            probs_list = []
            if p_incep is not None:
                probs_list.append(p_incep.reshape(1, -1))
            if p_rf and X_feat_scaled is not None:
                probs_list.append(p_rf.predict_proba(X_feat_scaled))
            if p_svm and X_feat_scaled is not None:
                probs_list.append(p_svm.predict_proba(X_feat_scaled))

            if probs_list:
                meta_X = np.hstack(probs_list)
                p_meta = models['meta'].predict_proba(meta_X)[0]
                idx = int(np.argmax(p_meta))
                print(" →", class_names[idx], f"(conf {p_meta[idx]:.4f})")
                print(format_probs(p_meta, class_names))
            else:
                print(" ⚠️ Not enough base probs for meta.")
        except Exception as e:
            print(" ⚠️ Meta predict error:", e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="xray_full_out", help="output folder (default xray_full_out)")
    p.add_argument("--image", help="optional image path to run single-image inference")
    args = p.parse_args()

    out_dir = Path(args.out)
    if not out_dir.exists():
        print("❌ Output folder not found:", out_dir)
        sys.exit(1)

    print_summary(out_dir)

    class_names = find_class_names(out_dir)
    print("\nClasses:", class_names if class_names else "❌ Not found")

    models = load_models_if_exist(out_dir)
    print("\n✅ Loaded models:", ", ".join(sorted(models.keys())))

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print("❌ Image not found:", img_path)
            sys.exit(1)
        infer_single_image(img_path, out_dir, class_names, models)
    else:
        print("\n(no image provided; use --image path/to/img.jpg)")
