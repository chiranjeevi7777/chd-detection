# Congenital Heart Disease (CHD) Chest X-Ray Classifier

An advanced AI-powered diagnostic system designed to classify Congenital Heart Diseases (CHD) from Chest X-ray images. This project implements a hybrid machine learning pipeline combining Deep Learning feature extraction (InceptionV3) with classical classifiers and ensemble stacking to maximize prediction accuracy and reliability.

## 🫀 Supported Classifications
- **ASD** (Atrial Septal Defect)
- **VSD** (Ventricular Septal Defect)
- **PDA** (Patent Ductus Arteriosus)
- **Normal** (Healthy Control)

---

## 🚀 Key Features
1. **Hybrid Ensemble Architecture**: Uses an InceptionV3 base model to extract 2048-dimensional chest X-ray features, then passes them to calibrated classical models (SVM, Random Forest, Decision Tree) and a Stacking Meta-Classifier.
2. **Interactive Streamlit Web UI**: Elegant frontend for uploading images, visualizing class probabilities, comparing models, and downloading detailed diagnosis reports.
3. **Comprehensive CLI Tools**: Quick inference scripts and pipeline tools for command-line prediction.
4. **Reproducible Training Pipeline**: Full script for training, feature extraction, calibration, cross-validation, and optional fine-tuning of the deep neural network.

---

## 📂 Project Structure

```
Chd_project/
├── data/                      # Dataset containing chest X-rays categorized by class
│   └── xray/
│       ├── ASD/
│       ├── Normal/
│       ├── PDA/
│       └── VSD/
├── inputs/                    # Sample X-ray images for testing
│   └── xray/
├── models/                    # Saved deep learning models
│   └── inceptionv3_model2.h5  # Pretrained primary InceptionV3 model
├── xray_full_out/             # Output folder for the trained classical/ensemble models
│   ├── svm.joblib
│   ├── random_forest.joblib
│   ├── meta_logistic.joblib
│   ├── scaler.joblib          # Saved StandardScaler for model compatibility
│   └── inception_head_model.h5
├── app2.py                    # Streamlit Web Application
├── detect.py                  # CLI script for primary Keras model inference
├── read.py                    # CLI script for comprehensive pipeline inference
├── train_full_pipeline.py     # End-to-end training and evaluation script
├── xray_manifest.csv          # CSV mapping images to labels
└── requirements.txt           # Project dependencies
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Chd_project
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 to 3.11 installed. Install required packages using:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Project

### 1. Interactive Web Interface (Streamlit)
To launch the beautiful web dashboard:
```bash
streamlit run app2.py
```
*Features:*
- Upload any Chest X-ray image (JPG, PNG, BMP).
- View side-by-side model predictions and probability charts.
- Download a detailed text report of the diagnostic predictions.

### 2. Quick CLI Inference (Primary Model)
To run a fast prediction using the primary InceptionV3 model:
```bash
python detect.py
```
*(By default, this runs on a sample image from the `inputs/xray` directory. You can edit the script to change the input path or run custom files.)*

### 3. Pipeline CLI Inference (Full Ensemble)
To evaluate an image against all models in the ensemble (Deep learning, SVM, Random Forest, Stacking Meta-Classifier) with tabular outputs:
```bash
python read.py --image inputs/xray/PDA01.jpg
```

---

## 🏋️ Training the Models

If you wish to retrain the classifiers and re-run the stacking pipeline on your data:
```bash
python train_full_pipeline.py --manifest xray_manifest.csv --out_dir xray_full_out --epochs_head 25 --kfold 5
```
*Pipeline Steps:*
1. Splits the dataset in a stratified manner (Train/Val/Test).
2. Extracts deep features from all images using a pretrained ImageNet InceptionV3 model.
3. Fits and calibrates classical classifiers (SVM, RF, Decision Tree) on the extracted features.
4. Trains a meta-classifier (Logistic Regression Stacking) combining the probabilities from all base estimators.
5. Saves classification reports, confusion matrices, and Grad-CAM visual explanations to the output folder.

---

## 📊 Methodology & Model Calibration

To prevent overconfidence, the project uses **Platt Scaling** (via `CalibratedClassifierCV`) on the classical models. This transforms raw decision values into true probability distributions that match deep learning outputs. The Meta Stacking model integrates:
- Deep neural network probabilities
- Calibrated Random Forest probabilities
- Calibrated Support Vector Machine probabilities

These inputs are combined via a **Meta-Logistic Regression** classifier to deliver a unified, highly robust diagnostic decision.
