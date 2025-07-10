# 🫁 Lung & Colon Cancer Detection using CNN and Transfer Learning

## 🧾 Project Overview
This project uses **Convolutional Neural Networks (CNN)** and **Transfer Learning (MobileNetV2)** to detect and classify medical images into cancerous and non-cancerous categories, specifically for lung and colon histopathological tissues. It also incorporates fine-tuning, data augmentation, model evaluation, and Grad-CAM-based explainability.

---

## 📁 Workflow Summary

### 1. 📦 Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- Format: JPEG histopathological images
- Categories:
  - `lung_aca` – Lung Adenocarcinoma
  - `lung_scc` – Lung Squamous Cell Carcinoma
  - `lung_n` – Normal Lung Tissue
  - `colon_aca` – Colon Adenocarcinoma
  - `colon_n` – Normal Colon Tissue

---

### 2. 🧰 Environment & Libraries
- Tools: `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `OpenCV`, `TensorFlow`, `Keras`, `scikit-learn`
- Image handling: Resizing, normalization, one-hot encoding, and splitting (`train/val`)

---

### 3. 🖼️ Data Visualization
- Random sample display for each class
- Class-wise image count summary
- Path validation checks for dataset integrity

---

### 4. 🧪 Preprocessing
- Resize to `128×128`
- One-hot label encoding
- Data split: `80/20` (Train/Val)
- ~200 images loaded per class
  - Total dataset size: ~1000 images

---

### 5. 🧠 Model Architecture

#### Option 1: Custom CNN
```text
Conv2D ×3 → MaxPooling → Flatten → Dense(256) → BatchNorm
→ Dense(128) → Dropout → Output(Softmax)
```

#### Option 2: Transfer Learning (MobileNetV2)
```text
MobileNetV2 (pretrained) → GlobalAvgPooling → Dense(128)
→ Dropout → BatchNorm → Output(Softmax)
```

---

### 6. 🔁 Training Strategy
- Optimizer: `Adam`
- Loss: `CategoricalCrossentropy`
- Epochs: `10`
- Batch Size: `32`
- Callbacks:
  - `EarlyStopping` (patience=3, restore best weights)
  - `ReduceLROnPlateau` (patience=2, factor=0.5)
  - Custom Callback: Stops training if `val_accuracy > 85%`

---

### 7. 🎛️ Fine-Tuning MobileNetV2
- Unfreeze last 10% of layers
- Freeze initial 90% to retain learned features
- Retrain with data augmentation
- Augmentations used:
  - Rotation, shift, shear, zoom, horizontal flip
  - Fill mode: `nearest`

---

## 📊 Evaluation & Visuals

### 📈 Training Performance
- Accuracy and loss graphs (`train/val`)

### 🧾 Classification Report
- Precision, Recall, F1-score via `sklearn.metrics`

### 🧮 Confusion Matrix
- Visualized with `seaborn.heatmap`

---

## 🔬 Explainability with Grad-CAM
- Highlights image regions contributing to predictions
- Overlay heatmaps on validation samples
- Visualized with `matplotlib` side-by-side
- Helps interpret model confidence for each prediction

---

## 💾 Save & Load Model
```python
model.save("fine_tuned_lung_colon_classifier.h5")
# Load later:
model = keras.models.load_model("fine_tuned_lung_colon_classifier.h5")
```

---

## 🚀 Future Enhancements
- Export to mobile-friendly formats (`TFLite`, `ONNX`)
- Improve class balance and boost generalization
- Use full dataset or expand training set
- Build interactive dashboard or web interface
- Schedule experiments with more advanced callbacks

---

## 📜 License & Acknowledgments
- Educational and research purposes only
- Dataset credited to original [Kaggle contributors](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
