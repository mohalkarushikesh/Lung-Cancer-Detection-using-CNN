# 🫁 Lung Cancer Detection using CNN

## 🧾 Description
A deep learning model using **Convolutional Neural Networks (CNN)** is developed to classify medical images into cancerous and non-cancerous categories. The model processes visual patterns found in histopathological scans or CT images to assist in faster and more reliable diagnosis of lung cancer.

---

## 📁 Project Workflow

### 1. 📦 Data Acquisition
- Utilizes [Kaggle dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) containing labeled `.jpeg` images of lung and colon tissues.
- Includes categories:
  - `lung_aca` – Lung Adenocarcinoma
  - `lung_scc` – Lung Squamous Cell Carcinoma
  - `lung_n` – Normal Lung Tissue
  - *(Optionally includes colon tissue classes)*

### 2. 📚 Importing Libraries
- Essential packages: `NumPy`, `Pandas`, `Matplotlib`, `OpenCV`, `TensorFlow`, `Keras`, `scikit-learn`
- Preprocessing: image resizing, class detection, and dataset balancing

### 3. 🖼️ Data Visualization
- Displays random sample images per class to confirm distribution and image quality
- Summarizes image count across each class

### 4. 🧼 Image Preprocessing
- Resizes images to uniform dimensions (`128x128`)
- Normalizes pixel values
- One-hot encodes class labels
- Splits dataset into training and validation sets (`80/20`)

---

## 🧠 Model Development

### 🛠️ Architecture Overview

#### ✅ Option 1: Custom CNN
```text
Input → [Conv2D → ReLU → MaxPooling] ×3 → Flatten → Dense(256)
→ BatchNorm → Dense(128) → Dropout(0.3) → Output (Softmax)
```

#### ✅ Option 2: Transfer Learning (MobileNetV2)
```text
Input → MobileNetV2 (frozen) → GlobalAveragePooling → Dense(128)
→ Dropout(0.3) → BatchNorm → Output (Softmax)
```

### 🔁 Training Setup
- Optimizer: `Adam`
- Loss Function: `Categorical Crossentropy`
- Batch Size: `32`
- Epochs: `10`
- Callbacks:
  - `EarlyStopping`: Stops training if validation accuracy plateaus
  - `ReduceLROnPlateau`: Dynamically adjusts learning rate
  - Custom callback for 90% validation accuracy threshold

---

## 🧪 Model Evaluation

### 📈 Performance Metrics
- **Accuracy and Loss curves** plotted across training epochs
- **Classification Report** with:
  - Precision
  - Recall
  - F1-score

### 🔍 Confusion Matrix
- Visual analysis of model predictions vs. actual labels
- Helps identify misclassified classes

---

## 💾 Saving & Reloading Model

```python
# Save trained model
model.save("lung_colon_classifier.h5")

# Load model later
model = keras.models.load_model("lung_colon_classifier.h5")
```

---

## 🚀 Future Improvements
- Enable fine-tuning of pretrained layers
- Apply data augmentation techniques (`ImageDataGenerator`)
- Incorporate Grad-CAM for visualizing model attention
- Deploy model with `TensorFlow Lite` or convert to `ONNX`

---

## 📜 License
This project is intended for educational and research use. The dataset is provided by Kaggle contributors under its license terms.
