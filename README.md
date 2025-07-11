# 🧬 Lung & Colon Histopathology Image Classifier

This project builds a convolutional neural network using transfer learning (MobileNetV2) to classify lung and colon tissue histopathology images. It leverages pre-trained models, extensive data preprocessing, and visualization tools to achieve high accuracy in identifying cancerous tissue types.

---

## 📁 Dataset Overview

- **Source**: `./lung_colon_image_set/`
- **Structure**:
  ```
  lung_colon_image_set/
    ├── lung_image_sets/
    │     ├── lung_aca
    │     ├── lung_n
    │     └── lung_scc
    ├── colon_image_sets/
          ├── colon_aca
          └── colon_n
  ```
- **Classes Detected**:  
  - `lung_aca` (Lung Adenocarcinoma)  
  - `lung_n` (Normal Lung)  
  - `lung_scc` (Lung Squamous Cell Carcinoma)  
  - `colon_aca` (Colon Adenocarcinoma)  
  - `colon_n` (Normal Colon)

---

## 🛠️ Project Pipeline

### 1. **Data Preparation**
- Loads `.jpeg` images from respective class folders
- Resizes to `128x128` using OpenCV
- Labels are one-hot encoded for multi-class classification
- Splits dataset into training and validation using `train_test_split`

### 2. **Model Architecture**
- Uses MobileNetV2 (pre-trained on ImageNet) as a base model
- Custom top layers include:
  - GlobalAveragePooling
  - Dense + Dropout + BatchNormalization
  - Final softmax classification layer
- Freezes base layers initially for transfer learning

### 3. **Training Strategy**
- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`
- Callbacks:
  - EarlyStopping (monitoring val_accuracy)
  - ReduceLROnPlateau (monitoring val_loss)
  - Custom callback to stop training at 90% validation accuracy

### 4. **Evaluation**
- Uses `sklearn.metrics.classification_report` and `confusion_matrix`
- Generates plots for training accuracy and loss over epochs

---

## 📊 Performance Visualization

- Training/Validation Accuracy and Loss curves are plotted using `matplotlib`
- Sample images are displayed per class for visual inspection
- Classification metrics like precision, recall, and F1-score reported

---

## 📦 Output

- Final trained model saved as:  
  ```
  lung_colon_classifier.h5
  ```

---

## 🚀 Future Enhancements

| Feature                                   | Description                                                                 |
|------------------------------------------|-----------------------------------------------------------------------------|
| 🔍 Grad-CAM Integration                  | Visualize class-specific activation maps to explain model decisions         |
| 🔄 Fine-Tuning Base Model                | Unfreeze deeper layers in MobileNetV2 for refined feature learning          |
| 📈 Class Imbalance Handling              | Apply `class_weight` or data resampling to balance minority classes         |
| 🧪 Test Set Evaluation                   | Add separate test split for unbiased performance validation                 |
| 📁 Image Logging                         | Save misclassified images for manual review and dataset improvement         |
| ⚙️ Hyperparameter Tuning                 | Use `KerasTuner` or grid search to find optimal architecture/configuration |
| 📂 Model Format Upgrade                  | Save in `.keras` format as recommended for modern serialization             |
| 📊 TensorBoard Logging                   | Track model metrics, images, and learning rate schedules interactively      |
| 📉 Model Compression                     | Apply quantization or pruning for deployment on edge devices                |

---

## 📚 Dependencies

```bash
tensorflow==2.x
opencv-python
matplotlib
scikit-learn
pandas
Pillow
```

---

## 👨‍⚕️ Use Case

- Histopathology image classification for research and diagnostics
- Can be adapted for real-time medical decision support
- Ideal starting point for pathology detection pipelines in healthcare ML
