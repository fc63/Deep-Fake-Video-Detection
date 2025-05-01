
# DeepFake Detection with CNNs & Transfer Learning

This assignment is part of the **CENG 481 - Artificial Neural Networks** course assignment.
It addresses the task of detecting deepfake content using image-based CNN classification and transfer learning techniques.

---

## 🎯 Objective

- Build an end-to-end image-based deepfake detection pipeline
- Extract and align 10 frames per video clip
- Pair each real frame with a corresponding fake variation
- Train a CNN model using EfficientNetB0 with ImageNet weights
- Apply regularization, checkpointing, and early stopping for best performance
- Evaluate using AUC-ROC, accuracy, precision, recall, F1-score

---

## 📦 Dataset

- **Source**: [DFDC Part-34 on Kaggle](https://www.kaggle.com/datasets/greatgamedota/dfdc-part-34)
- **Metadata**: `metadata34.csv`
- Each video is represented by 10 frames: `0.jpg`, `30.jpg`, ..., `270.jpg`
- Fake videos are linked to their originals via metadata

---

## 🧠 Model

- Base: `EfficientNetB0`, pretrained on ImageNet
- Frozen base trained with custom head; then base unfrozen and fine-tuned
- Architecture: GlobalAveragePooling2D → Dropout(0.4) → Dense(1, sigmoid)
- Input size: 224×224×3
- Optimizer: Adam (`lr=1e-4` frozen, `lr=1e-5` unfrozen)
- Loss: Binary Crossentropy
- Metrics: AUC, Accuracy, Precision, Recall, F1

---

## 🏋️ Training

- Balanced dataset from 6784 images (REAL + FAKE)
- Train/Test split: 79% / 21% (stratified)
- Batch size: 32
- Epochs: max 100 (early stopping with patience=8)
- Model checkpointing enabled (.keras format)
- TensorBoard used for experiment tracking
- Platform: Google Colab (GPU)

---

## 🧪 Evaluation (Final Results)

- Accuracy: 0.80
- AUC-ROC: 0.88
- Precision: 0.78
- Recall: 0.82
- F1-Score: 0.80

---

## 💾 How to Use

```python
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load and preprocess image
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img.astype(np.float32)

# Download and load model
model_path = hf_hub_download(repo_id="fc63/deepfake-detection-cnn_v2", filename="best_model.keras")
model = load_model(model_path)

# Predict
img = preprocess_image("frame.jpg")
pred = model.predict(img[np.newaxis, ...])

print("FAKE" if pred[0][0] > 0.5 else "REAL")
```

---

## 📁 Requirements

```
tensorflow
scikit-learn
pandas
matplotlib
opencv-python
huggingface_hub
```

---

## 🔗 Repositories

- 🤗 Model: https://huggingface.co/fc63/deepfake-detection-cnn_v2
- 💻 Codebase: https://github.com/fc63/Deep-Fake-Video-Detection

---

## ⚠️ Ethical Considerations

Deepfake technology poses threats to media trust, privacy, and security. This assignment aims to mitigate misuse by improving detection accuracy while acknowledging dataset limitations and the risk of bias.

---

## 👤 Author

**Furkan Çoban**  
Çankaya University

---

## 🧑‍🏫 Instructor

This assignment was completed as part of the CENG 481 - Artificial Neural Networks course  
at Çankaya University under the supervision of **Dr. Nurdan Saran**.
