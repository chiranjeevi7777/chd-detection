#!/usr/bin/env python3
"""
train_full_pipeline.py

Full pipeline:
 - Read manifest CSV (filepath,label)
 - Stratified split: train / val / test
 - Feature extraction using InceptionV3 (include_top=False, pooling='avg') -> saves .npy
 - StandardScaler for classical classifiers
 - Train classical models:
     - DecisionTree (baseline)
     - RandomForest (GridSearch), SVM (GridSearch)
   -> optional probability calibration (CalibratedClassifierCV)
 - Train Keras dense head on features (EarlyStopping)
 - k-fold stacking (OOF) to train meta-learner (LogisticRegression)
 - Evaluate all models on test set: accuracy, precision, recall, f1, confusion matrix
 - Optionally fine-tune top Inception blocks using ImageDataGenerator (augmentation)
 - Produce Grad-CAM maps for some test images (using fine-tuned or head model)
 - Save models, scalers, probs, metrics, confusion matrices and plots

Author: ChatGPT
"""
import argparse, os, sys, math, json
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- helpers ----------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def read_manifest(manifest_path):
    df = pd.read_csv(manifest_path)
    if 'filepath' not in df.columns or 'label' not in df.columns:
        raise ValueError("Manifest must contain 'filepath' and 'label' columns.")
    
    resolved_paths = []
    labels = []
    missing_count = 0
    
    for idx, row in df.iterrows():
        path = row['filepath']
        label = row['label']
        
        if os.path.exists(path):
            resolved_paths.append(path)
            labels.append(label)
            continue
            
        filename = os.path.basename(path)
        rel_path = os.path.join('data', 'xray', label, filename)
        if os.path.exists(rel_path):
            resolved_paths.append(rel_path)
            labels.append(label)
            continue
            
        missing_count += 1
        
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} files listed in manifest could not be found and were skipped.")
        
    return pd.DataFrame({'filepath': resolved_paths, 'label': labels})

def stratified_split(df, seed=42, test_size=0.15, val_size=0.10):
    labels = df['label'].values
    trainval_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size, stratify=labels, random_state=seed)
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    # split trainval into train and val with proportion val_size/(1-test_size) of full
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(np.arange(len(trainval)), test_size=rel_val, stratify=trainval['label'].values, random_state=seed+1)
    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)
    return train, val, test

def build_inception_extractor(input_shape=(299,299,3)):
    base = applications.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    preprocess = applications.inception_v3.preprocess_input
    return base, preprocess

def extract_features_for_list(paths, extractor, preprocess_fn, target_size=(299,299), batch=32):
    n = len(paths)
    feats = []
    for i in tqdm(range(0, n, batch), desc="Extracting features"):
        batch_paths = paths[i:i+batch]
        imgs = []
        for p in batch_paths:
            arr = tf.keras.utils.load_img(p, target_size=target_size)
            arr = tf.keras.utils.img_to_array(arr)
            arr = preprocess_fn(arr)
            imgs.append(arr)
        batch_arr = np.stack(imgs, axis=0)
        batch_feats = extractor.predict(batch_arr, verbose=0)
        feats.append(batch_feats)
    feats = np.vstack(feats) if feats else np.zeros((0, extractor.output_shape[1]))
    return feats

def build_keras_head(input_dim, n_classes, lr=1e-3):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dropout(0.4)(inp)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def eval_metrics(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy":acc, "precision_macro":prec, "recall_macro":rec, "f1_macro":f1, "report":report, "cm":cm}

def save_report_and_cm(out_dir, prefix, metrics, class_names):
    txt = f"{prefix} metrics\nAccuracy: {metrics['accuracy']}\nPrecision(macro): {metrics['precision_macro']}\nRecall(macro): {metrics['recall_macro']}\nF1(macro): {metrics['f1_macro']}\n\n{metrics['report']}"
    Path(out_dir, f"{prefix}_results_report.txt").write_text(txt)
    cm_df = pd.DataFrame(metrics['cm'], index=class_names, columns=class_names)
    cm_df.to_csv(Path(out_dir, f"{prefix}_results_confusion_matrix.csv"))

def plot_cm_and_save(cm, class_names, out_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# ---------- pipeline main ----------
def main(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    manifest = read_manifest(args.manifest)
    ensure_dir(args.out_dir)

    # split
    train_df, val_df, test_df = stratified_split(manifest, seed=args.seed, test_size=args.test_size, val_size=args.val_size)
    print(f"Classes: {sorted(manifest['label'].unique())}")
    print("Sizes: train", len(train_df), "val", len(val_df), "test", len(test_df))

    # save split manifests for reproducibility
    train_df.to_csv(Path(args.out_dir, "train_manifest.csv"), index=False)
    val_df.to_csv(Path(args.out_dir, "val_manifest.csv"), index=False)
    test_df.to_csv(Path(args.out_dir, "test_manifest.csv"), index=False)

    # label encoding
    le = LabelEncoder()
    le.fit(manifest['label'].values)
    class_names = list(le.classes_)
    print("Class names:", class_names)

    # build extractor
    extractor, preprocess_fn = build_inception_extractor(input_shape=(299,299,3))
    feat_dim = extractor.output_shape[1]
    print("Feature dim:", feat_dim)

    # extract features (saved if not exist)
    feat_train_path = Path(args.out_dir)/"features_train.npy"
    feat_val_path = Path(args.out_dir)/"features_val.npy"
    feat_test_path = Path(args.out_dir)/"features_test.npy"
    if not feat_train_path.exists() or args.force_extract:
        X_train_feats = extract_features_for_list(train_df['filepath'].tolist(), extractor, preprocess_fn, target_size=(299,299), batch=args.batch)
        np.save(feat_train_path, X_train_feats)
    else:
        X_train_feats = np.load(feat_train_path)
    if not feat_val_path.exists() or args.force_extract:
        X_val_feats = extract_features_for_list(val_df['filepath'].tolist(), extractor, preprocess_fn, target_size=(299,299), batch=args.batch)
        np.save(feat_val_path, X_val_feats)
    else:
        X_val_feats = np.load(feat_val_path)
    if not feat_test_path.exists() or args.force_extract:
        X_test_feats = extract_features_for_list(test_df['filepath'].tolist(), extractor, preprocess_fn, target_size=(299,299), batch=args.batch)
        np.save(feat_test_path, X_test_feats)
    else:
        X_test_feats = np.load(feat_test_path)

    y_train = le.transform(train_df['label'].values)
    y_val = le.transform(val_df['label'].values)
    y_test = le.transform(test_df['label'].values)
    np.save(Path(args.out_dir,"y_train.npy"), y_train)
    np.save(Path(args.out_dir,"y_val.npy"), y_val)
    np.save(Path(args.out_dir,"y_test.npy"), y_test)

    # standardize features for classical models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feats)
    X_val_scaled = scaler.transform(X_val_feats)
    X_test_scaled = scaler.transform(X_test_feats)
    joblib.dump(scaler, Path(args.out_dir,"scaler.joblib"))

    # ---------- classical models ----------
    results = {}

    # Decision Tree (quick baseline)
    dt = DecisionTreeClassifier(random_state=args.seed, max_depth=args.dt_max_depth)
    dt.fit(X_train_scaled, y_train)
    y_pred_dt = dt.predict(X_test_scaled)
    metrics_dt = eval_metrics(y_test, y_pred_dt, class_names)
    save_report_and_cm(args.out_dir, "decision_tree", metrics_dt, class_names)
    joblib.dump(dt, Path(args.out_dir,"decision_tree.joblib"))
    results['decision_tree'] = metrics_dt
    print("Decision Tree done.")

    # Random Forest (GridSearch)
    print("GridSearch RandomForest (this may take some time)...")
    rf_base = RandomForestClassifier(random_state=args.seed, n_jobs=-1)
    rf_params = {"n_estimators":[100,200], "max_depth":[None,20], "max_features":["sqrt","log2"]}
    rf_gs = GridSearchCV(rf_base, rf_params, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    rf_gs.fit(X_train_scaled, y_train)
    print("RF best:", rf_gs.best_params_, "score:", rf_gs.best_score_)
    rf = rf_gs.best_estimator_
    joblib.dump(rf, Path(args.out_dir,"random_forest.joblib"))
    y_pred_rf = rf.predict(X_test_scaled)
    metrics_rf = eval_metrics(y_test, y_pred_rf, class_names)
    save_report_and_cm(args.out_dir, "random_forest", metrics_rf, class_names)
    results['random_forest'] = metrics_rf

    # SVM (GridSearch)
    print("GridSearch SVM (may be slow)...")
    svc_base = SVC(probability=True, random_state=args.seed)
    svc_params = {"C":[0.1,1,5], "gamma":["scale","auto"], "kernel":["rbf"]}
    svc_gs = GridSearchCV(svc_base, svc_params, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    svc_gs.fit(X_train_scaled, y_train)
    print("SVM best:", svc_gs.best_params_, "score:", svc_gs.best_score_)
    svm = svc_gs.best_estimator_
    joblib.dump(svm, Path(args.out_dir,"svm.joblib"))
    y_pred_svm = svm.predict(X_test_scaled)
    metrics_svm = eval_metrics(y_test, y_pred_svm, class_names)
    save_report_and_cm(args.out_dir, "svm", metrics_svm, class_names)
    results['svm'] = metrics_svm

    # Calibrate RF and SVM using validation set
    print("Calibrating RF and SVM using validation set (isotonic)...")
    calib_rf = CalibratedClassifierCV(rf, cv='prefit', method='isotonic')
    calib_rf.fit(X_val_scaled, y_val)
    joblib.dump(calib_rf, Path(args.out_dir,"random_forest_calibrated.joblib"))
    calib_svm = CalibratedClassifierCV(svm, cv='prefit', method='isotonic')
    calib_svm.fit(X_val_scaled, y_val)
    joblib.dump(calib_svm, Path(args.out_dir,"svm_calibrated.joblib"))

    # ---------- Keras head training on features ----------
    keras_head = build_keras_head(feat_dim, n_classes=len(class_names), lr=args.lr_head)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = keras_head.fit(X_train_feats, y_train, validation_data=(X_val_feats, y_val),
                             epochs=args.epochs_head, batch_size=args.batch, callbacks=[es], verbose=2)
    keras_head.save(Path(args.out_dir,"inception_head_model.h5"))
    # evaluate
    probs_incep_test = keras_head.predict(X_test_feats, batch_size=args.batch)
    y_pred_incep = np.argmax(probs_incep_test, axis=1)
    metrics_incep = eval_metrics(y_test, y_pred_incep, class_names)
    save_report_and_cm(args.out_dir, "inception_head", metrics_incep, class_names)
    np.save(Path(args.out_dir,"probs_inception_test.npy"), probs_incep_test)
    results['inception_head'] = metrics_incep

    # ---------- k-fold stacking (OOF) ----------
    k = args.kfold
    print(f"Starting {k}-fold stacking to build OOF probs for meta-learner (this will retrain models k times)...")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
    n_train = len(X_train_feats)
    # placeholders for OOF probabilities
    oof_rf = np.zeros((n_train, len(class_names)))
    oof_svm = np.zeros((n_train, len(class_names)))
    oof_incep = np.zeros((n_train, len(class_names)))

    X_train_all = X_train_feats.copy()
    y_train_all = y_train.copy()

    fold_idx = 0
    for train_idx, val_idx in skf.split(X_train_all, y_train_all):
        fold_idx += 1
        print(f"Fold {fold_idx}/{k}")
        X_tr, X_val_fold = X_train_all[train_idx], X_train_all[val_idx]
        y_tr, y_val_fold = y_train_all[train_idx], y_train_all[val_idx]

        # scale
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr); X_val_s = sc.transform(X_val_fold)

        # RF
        rf_fold = RandomForestClassifier(**rf.get_params())
        rf_fold.fit(X_tr_s, y_tr)
        oof_rf[val_idx] = rf_fold.predict_proba(X_val_s)

        # SVM
        svm_fold = SVC(**{k:v for k,v in svm.get_params().items() if k!='probability'}, probability=True)
        svm_fold.set_params(**{k:v for k,v in svm.get_params().items() if k!='probability'})
        svm_fold.fit(X_tr_s, y_tr)
        oof_svm[val_idx] = svm_fold.predict_proba(X_val_s)

        # Inception head (train small head on features)
        head_fold = build_keras_head(feat_dim, n_classes=len(class_names), lr=args.lr_head)
        es2 = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
        head_fold.fit(X_tr, y_tr, validation_data=(X_val_fold, y_val_fold), epochs=10, batch_size=args.batch, callbacks=[es2], verbose=0)
        oof_incep[val_idx] = head_fold.predict(X_val_fold, batch_size=args.batch)

    # Save OOF probs
    np.save(Path(args.out_dir,"oof_rf.npy"), oof_rf)
    np.save(Path(args.out_dir,"oof_svm.npy"), oof_svm)
    np.save(Path(args.out_dir,"oof_incep.npy"), oof_incep)

    # Train meta-learner (LogisticRegression) on OOF probs
    meta_X = np.hstack([oof_incep, oof_rf, oof_svm])
    meta_y = y_train_all
    meta = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    meta.fit(meta_X, meta_y)
    joblib.dump(meta, Path(args.out_dir,"meta_logistic.joblib"))

    # ---------- Evaluate final ensembles on the test set ----------
    # prepare base model probs on test (trained on full train set)
    rf_final = rf
    svm_final = svm
    # scale using saved scaler
    X_test_s = scaler.transform(X_test_feats)
    probs_rf_test = rf_final.predict_proba(X_test_s)
    probs_svm_test = svm_final.predict_proba(X_test_s)
    probs_incep_test = probs_incep_test  # already computed above

    np.save(Path(args.out_dir,"probs_rf_test.npy"), probs_rf_test)
    np.save(Path(args.out_dir,"probs_svm_test.npy"), probs_svm_test)
    # avg fusion (equal weights)
    avg_probs = (probs_incep_test + probs_rf_test + probs_svm_test) / 3.0
    y_pred_avg = np.argmax(avg_probs, axis=1)
    metrics_avg = eval_metrics(y_test, y_pred_avg, class_names); save_report_and_cm(args.out_dir, "fusion_avg_all", metrics_avg, class_names)
    np.save(Path(args.out_dir,"probs_avg_all.npy"), avg_probs)
    results['fusion_avg_all'] = metrics_avg

    # stacking meta
    meta_test_X = np.hstack([probs_incep_test, probs_rf_test, probs_svm_test])
    probs_meta_test = meta.predict_proba(meta_test_X)
    y_pred_meta = np.argmax(probs_meta_test, axis=1)
    metrics_meta = eval_metrics(y_test, y_pred_meta, class_names); save_report_and_cm(args.out_dir, "fusion_stacking_meta", metrics_meta, class_names)
    np.save(Path(args.out_dir,"probs_meta_test.npy"), probs_meta_test)
    joblib.dump(meta, Path(args.out_dir,"meta_logistic.joblib"))
    results['fusion_stacking_meta'] = metrics_meta

    # ---------- Optional fine-tune Inception (image-level) ----------
    if args.fine_tune and args.fine_tune_epochs > 0:
        print("Starting fine-tuning of top Inception blocks with data augmentation...")
        # build full model (base + head) and initialize head weights from trained keras_head
        base = applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299,299,3), pooling='avg')
        # attach new head
        x = base.output
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        out = layers.Dense(len(class_names), activation='softmax')(x)
        full_model = models.Model(base.input, out)
        # set head weights from keras_head (dense layers sequentially)
        try:
            # keras_head structure: Input -> Dropout -> Dense(256) -> Dropout -> Dense(n)
            head_weights = [l.get_weights() for l in keras_head.layers if isinstance(l, layers.Dense) or isinstance(l, layers.Dropout)]
            # We will set weights for the final two Dense layers in full_model by matching shapes
            # find Dense(256) and Dense(n) layers in full_model
            dense_layers = [l for l in full_model.layers if isinstance(l, layers.Dense)]
            if len(dense_layers) >= 2:
                # assign weights of keras_head dense layers to these
                dense_layers[-2].set_weights(keras_head.layers[2].get_weights())  # dense 256 is layer index 2 in build_keras_head
                dense_layers[-1].set_weights(keras_head.layers[4].get_weights())
                print("Transferred head weights to full_model.")
        except Exception as e:
            print("Could not transfer head weights:", e)

        # freeze all then unfreeze last N layers
        for layer in base.layers:
            layer.trainable = False
        unfreeze_n = args.unfreeze_last
        if unfreeze_n > 0:
            for layer in base.layers[-unfreeze_n:]:
                layer.trainable = True
        # compile low LR
        full_model.compile(optimizer=tf.keras.optimizers.Adam(args.fine_tune_lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # build ImageDataGenerators using train_manifest and val_manifest
        datagen_train = ImageDataGenerator(rotation_range=12, width_shift_range=0.06, height_shift_range=0.06,
                                           shear_range=0.06, zoom_range=0.08, horizontal_flip=True, rescale=1./255)
        datagen_val = ImageDataGenerator(rescale=1./255)
        train_df2 = train_df.copy(); train_df2['label'] = le.inverse_transform(y_train)
        val_df2 = val_df.copy(); val_df2['label'] = le.inverse_transform(y_val)
        flow_train = datagen_train.flow_from_dataframe(train_df2, x_col='filepath', y_col='label', target_size=(299,299), class_mode='sparse', batch_size=args.batch, shuffle=True)
        flow_val = datagen_val.flow_from_dataframe(val_df2, x_col='filepath', y_col='label', target_size=(299,299), class_mode='sparse', batch_size=args.batch, shuffle=False)
        es_ft = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        full_model.fit(flow_train, validation_data=flow_val, epochs=args.fine_tune_epochs, callbacks=[es_ft], verbose=2)
        full_model.save(Path(args.out_dir,"inception_finetuned.keras"))
        # evaluate finetuned model on test images (use generator)
        datagen_test = ImageDataGenerator(rescale=1./255)
        test_df2 = test_df.copy(); test_df2['label'] = le.inverse_transform(y_test)
        flow_test = datagen_test.flow_from_dataframe(test_df2, x_col='filepath', y_col='label', target_size=(299,299), class_mode='sparse', batch_size=args.batch, shuffle=False)
        preds_ft = full_model.predict(flow_test, verbose=0)
        y_pred_ft = np.argmax(preds_ft, axis=1)
        metrics_ft = eval_metrics(y_test, y_pred_ft, class_names)
        save_report_and_cm(args.out_dir, "inception_finetuned", metrics_ft, class_names)
        results['inception_finetuned'] = metrics_ft

        # Grad-CAM for a few test images using finetuned model
        try:
            make_gradcam_and_save(full_model, flow_test, test_df2, class_names, args.out_dir, top_k=6)
        except Exception as e:
            print("Grad-CAM generation failed:", e)

    # Save summary metrics CSV
    summary = {k: {"accuracy":v["accuracy"], "precision_macro":v["precision_macro"], "recall_macro":v["recall_macro"], "f1_macro":v["f1_macro"]} for k,v in results.items()}
    pd.DataFrame(summary).T.to_csv(Path(args.out_dir,"models_summary_metrics.csv"))
    print("Pipeline finished. Outputs saved to:", args.out_dir)

# Grad-CAM helper (uses a trained keras model that accepts (299,299,3) images)
def make_gradcam_and_save(keras_model, generator, test_df, class_names, out_dir, top_k=6):
    ensure_dir(out_dir)
    import tensorflow as tf
    # find last conv layer name
    last_conv = None
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name; break
    if last_conv is None:
        raise RuntimeError("No Conv2D layer found for Grad-CAM.")
    print("Using last conv layer:", last_conv)
    # pick top_k misclassified or random images
    preds = keras_model.predict(generator, verbose=0)
    y_true = generator.classes
    y_pred = np.argmax(preds, axis=1)
    mis_idx = [i for i,(t,p) in enumerate(zip(y_true, y_pred)) if t!=p]
    sel = mis_idx[:top_k] if len(mis_idx)>=top_k else list(range(min(top_k,len(y_true))))
    for idx in sel:
        img_path = generator.filepaths[idx]
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299,299))
        arr = tf.keras.preprocessing.image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)/255.0
        # Grad-CAM implementation
        grad_model = tf.keras.models.Model([keras_model.inputs], [keras_model.get_layer(last_conv).output, keras_model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(arr)
            pred_index = np.argmax(predictions[0])
            loss = predictions[:, pred_index]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-9)
        # overlay
        import cv2
        img_orig = cv2.imread(img_path)
        img_orig = cv2.resize(img_orig, (299,299))
        heatmap = cv2.resize(heatmap, (299,299))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlayed = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)
        outp = Path(out_dir)/f"gradcam_{idx}_{Path(img_path).stem}.jpg"
        cv2.imwrite(str(outp), overlayed)
    print("Saved Grad-CAM overlays to", out_dir)

# ----------------- CLI -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV manifest with columns 'filepath','label'")
    p.add_argument("--out_dir", required=True, help="output folder")
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs_head", type=int, default=25)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rf_n_estimators", type=int, default=200)
    p.add_argument("--dt_max_depth", type=int, default=None)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--force_extract", action='store_true')
    p.add_argument("--fine_tune", action='store_true', help="Enable image-level fine-tuning")
    p.add_argument("--fine_tune_epochs", type=int, default=8)
    p.add_argument("--unfreeze_last", type=int, default=40, help="Unfreeze last N layers of inception for fine-tune")
    p.add_argument("--fine_tune_lr", type=float, default=1e-5)
    args = p.parse_args()
    # create out_dir
    ensure_dir(args.out_dir)
    # run
    main(args)
