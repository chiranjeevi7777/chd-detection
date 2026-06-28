# streamlit_xray_fusion.py
import streamlit as st
import tempfile, os, time
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input


import detect  # must provide load_model, predict_chd, CLASS_NAMES
from read import load_models_if_exist, find_class_names  # reuse your helper functions

st.set_page_config(page_title="CHD X-Ray Classifier", layout="wide", page_icon="🫀")
st.title("CHD X-Ray Classifier — ASD / VSD / PDA / Normal")

# Sidebar
st.sidebar.header("Model")
use_uploaded = st.sidebar.checkbox("Upload custom X-ray .h5 model", value=False)
uploaded_file = st.sidebar.file_uploader("Upload Keras .h5 model", type=["h5", "keras"]) if use_uploaded else None

out_dir = Path("xray_full_out")  # where classical models & head are stored
class_names = find_class_names(out_dir) or ["ASD", "VSD", "PDA", "Normal"]
models = load_models_if_exist(out_dir)

# Load model
@st.cache_resource(show_spinner=False)
def load_xray_model(uploaded_model_file):
    if uploaded_model_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        tmp.write(uploaded_model_file.read()); tmp.flush(); tmp.close()
        return detect.load_model(tmp.name), tmp.name
    else:
        return detect.load_model(getattr(detect, "MODEL_PATH")), getattr(detect, "MODEL_PATH")

xray_model, xray_model_src = None, None
try:
    xray_model, xray_model_src = load_xray_model(uploaded_file) if (use_uploaded and uploaded_file) else load_xray_model(None)
    st.sidebar.success(f"Loaded model: {Path(xray_model_src).name}")
except Exception as e:
    st.sidebar.error(f"Model load failed: {e}")

# Upload image
uploaded_img = st.file_uploader("Upload X-ray image (jpg/png)", type=["jpg","jpeg","png","bmp"])
if uploaded_img is None:
    st.info("Upload an image to run prediction.")
    st.stop()

# Save temp
tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_img.name).suffix)
tmp_img.write(uploaded_img.read()); tmp_img.flush(); tmp_img.close()
img_path = tmp_img.name

# Inference helpers
def run_head(model, arr_proc):
    raw = model.predict(arr_proc)
    probs = np.asarray(raw[0], dtype=float)
    # Check if the output is already a probability distribution (sums to 1.0)
    if not np.allclose(np.sum(probs), 1.0, atol=1e-3):
        ex = np.exp(probs - np.max(probs))
        probs = ex / ex.sum()
    idx = int(np.argmax(probs))
    return class_names[idx], probs[idx], probs

def run_classical(model, feat, name):
    try:
        probs = model.predict_proba(feat)[0]
        idx = int(np.argmax(probs))
        return class_names[idx], probs[idx], probs
    except Exception:
        pred = model.predict(feat)[0]
        # Return probability vector
        if hasattr(model, "classes_"):
            v = np.zeros(len(class_names))
            try:
                v[pred] = 1.0
            except Exception:
                pass
            return class_names[pred], 1.0, v
        return class_names[pred], 1.0, np.eye(len(class_names))[pred]

def prob_table(probs):
    return pd.DataFrame({"class": class_names, "probability": probs}).sort_values("probability", ascending=False)

# Preprocess
pil_img = Image.open(img_path).convert("RGB")
pil_resized = pil_img.resize((299, 299))
arr = np.asarray(pil_resized).astype("float32")/255.0
arr_proc = arr[np.newaxis, ...]

# Extract features for classical models
extractor = InceptionV3(include_top=False, weights="imagenet", pooling="avg", input_shape=(299,299,3))
feat = extractor.predict(arr_proc, verbose=0)

# Scale features if scaler is loaded
feat_scaled = models['scaler'].transform(feat) if 'scaler' in models else feat

# ---------------- Main prediction (Head) ----------------
with st.spinner("Running predictions..."):
    head_label, head_conf, head_probs = run_head(models['head'], feat)
    main_df = prob_table(head_probs)

st.subheader("Primary Prediction (Inception Head/ Xception Head)")
cols = st.columns([1,2])
with cols[0]:
    st.image(pil_img, caption="Uploaded X-ray", use_container_width=True)
with cols[1]:
    st.metric("Predicted Label", head_label, f"{head_conf*100:.2f}%")
    st.dataframe(main_df.style.format({"probability":"{:.4f}"}))
    st.bar_chart(pd.Series(head_probs, index=class_names))

# ---------------- Comparisons ----------------
st.subheader("Model Comparisons")
for name in ["random_forest_calibrated", "random_forest", "svm_calibrated", "svm", "decision_tree", "meta"]:
    if name in models:
        with st.expander(name.upper(), expanded=False):
            if name == "meta":
                # Meta stacking
                base_probs = []
                if "head" in models: base_probs.append(head_probs.reshape(1,-1))
                if "random_forest_calibrated" in models: base_probs.append(models["random_forest_calibrated"].predict_proba(feat_scaled))
                elif "random_forest" in models: base_probs.append(models["random_forest"].predict_proba(feat_scaled))
                if "svm_calibrated" in models: base_probs.append(models["svm_calibrated"].predict_proba(feat_scaled))
                elif "svm" in models: base_probs.append(models["svm"].predict_proba(feat_scaled))
                
                if base_probs:
                    meta_X = np.hstack(base_probs)
                    probs = models["meta"].predict_proba(meta_X)[0]
                    idx = int(np.argmax(probs))
                    pred, conf = class_names[idx], probs[idx]
                else:
                    pred, conf, probs = "N/A", 0.0, np.zeros(len(class_names))
            else:
                pred, conf, probs = run_classical(models[name], feat_scaled, name)
            df = prob_table(probs)
            st.metric("Prediction", pred, f"{conf*100:.2f}%")
            st.dataframe(df.style.format({"probability":"{:.4f}"}))
            st.bar_chart(pd.Series(probs, index=class_names))
# ---------------- Global Comparison & Conclusion ----------------
st.subheader("Final Conclusion")

# Save results dictionary for reporting
all_results = {}
all_results["head"] = (head_label, head_conf, head_probs)

# Collect classical + meta for comparison only
for name in ["random_forest_calibrated", "random_forest", "svm_calibrated", "svm", "decision_tree", "meta"]:
    if name in models:
        if name == "meta":
            base_probs = []
            if "head" in models: base_probs.append(head_probs.reshape(1,-1))
            if "random_forest_calibrated" in models: base_probs.append(models["random_forest_calibrated"].predict_proba(feat_scaled))
            elif "random_forest" in models: base_probs.append(models["random_forest"].predict_proba(feat_scaled))
            if "svm_calibrated" in models: base_probs.append(models["svm_calibrated"].predict_proba(feat_scaled))
            elif "svm" in models: base_probs.append(models["svm"].predict_proba(feat_scaled))
            
            if base_probs:
                meta_X = np.hstack(base_probs)
                probs = models["meta"].predict_proba(meta_X)[0]
                idx = int(np.argmax(probs))
                pred, conf = class_names[idx], probs[idx]
            else:
                pred, conf, probs = "N/A", 0.0, np.zeros(len(class_names))
        else:
            pred, conf, probs = run_classical(models[name], feat_scaled, name)
        all_results[name] = (pred, conf, probs)

# Final decision = Inception head
final_pred, final_conf, final_probs = all_results["head"]
conclusion = f"Primary Model suggests **{final_pred}** with {final_conf*100:.2f}% confidence."

st.success(conclusion)

# ---------------- Downloadable Report ----------------
lines = []
lines.append("===== CHD X-ray Report =====")
lines.append(f"Image: {Path(uploaded_img.name).name}")
lines.append("")
for model_name, (pred, conf, probs) in all_results.items():
    lines.append(f"[{model_name.upper()}]")
    lines.append(f"  Prediction: {pred} ({conf*100:.2f}%)")
    for cls, p in zip(class_names, probs):
        lines.append(f"    {cls}: {p*100:.2f}%")
    lines.append("")
lines.append("==== Final Conclusion ====")
lines.append(conclusion)

report_txt = "\n".join(lines)
st.download_button("Download full report", report_txt, file_name="chd_xray_report.txt", mime="text/plain")
