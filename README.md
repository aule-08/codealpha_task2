# 🔢 Handwritten Digit Recognition using CNN

A deep learning project implementing a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset.

**Final Test Accuracy: 99.14%** ✅

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Results](#results)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Project Overview

This project demonstrates the implementation of a simple yet effective Convolutional Neural Network for handwritten digit recognition. The model is trained on the famous MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9).

### Key Highlights

✅ **High Accuracy:** Achieved 99.14% test accuracy  
✅ **Simple Architecture:** Only 2 convolutional layers with 225,034 parameters  
✅ **Fast Training:** Completes training in ~4-5 minutes on CPU  
✅ **Well-Organized:** Structured codebase with proper documentation  
✅ **Reproducible:** Complete setup instructions included  

---

## 📊 Results

### Model Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 99.08% | 99.08% | **99.14%** |
| **Loss** | 0.0298 | 0.0339 | 0.0248 |

### Training Details

- **Total Epochs:** 10
- **Batch Size:** 128
- **Training Samples:** 48,000 (60,000 with 20% validation split)
- **Test Samples:** 10,000
- **Training Time:** ~4.5 minutes (CPU - Intel processor)
- **Model Size:** 879.04 KB

### Per-Class Performance

| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.99 | 0.99 | 0.99 | 980 |
| 1 | 0.99 | 1.00 | 1.00 | 1135 |
| 2 | 0.99 | 0.99 | 0.99 | 1032 |
| 3 | 0.99 | 0.99 | 0.99 | 1010 |
| 4 | 0.99 | 1.00 | 1.00 | 982 |
| 5 | 0.98 | 0.99 | 0.99 | 892 |
| 6 | 0.99 | 0.99 | 0.99 | 958 |
| 7 | 0.99 | 0.99 | 0.99 | 1028 |
| 8 | 0.99 | 0.99 | 0.99 | 974 |
| 9 | 1.00 | 0.98 | 0.99 | 1009 |

**Overall Accuracy:** 99.14%

---

## ✨ Features

### Core Functionality

- **Data Preprocessing:** Automatic normalization and reshaping
- **CNN Architecture:** 2 convolutional layers with max pooling
- **Training Pipeline:** Complete training with validation split
- **Model Evaluation:** Comprehensive metrics and visualizations
- **Model Persistence:** Save/load trained models (.h5 format)
- **Custom Prediction:** Predict digits from custom images

### Visualizations

- ✅ Sample MNIST digits display
- ✅ Training history (accuracy & loss curves)
- ✅ Prediction results with confidence scores
- ✅ Confusion matrix heatmap
- ✅ Classification report

### Organization

- ✅ Modular code structure
- ✅ Automatic directory creation
- ✅ Organized output files (models, plots, reports)
- ✅ Detailed logging and progress indicators

---

## 🏗️ Project Structure

```
Task 2/
│
├── models/                           # Saved model files
│   └── cnn_mnist.h5          # Trained model (879 KB)
│
├── results/                          # Training results
│   ├── plots/                        # Visualization plots
│   │   ├── 01_training_history.png
│   │   └── 02_confusion_matrix.png
│   │
│   └── reports/                      # Performance reports
│       ├── classification_report.txt
│       ├── model_summary.txt
│       └── training_results.txt
│
├── main.py                           # Main training script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10
- Miniconda or Anaconda
- Windows/Linux/macOS

### Step 1: Clone or Download Project

```bash
# Download the project files to your local machine
cd "F:\kushan\Task 2"
```

### Step 2: Create Conda Environment

```bash
# Create new environment with Python 3.10
conda create -n task2 python=3.10 -y

# Activate environment
conda activate task2
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Dependencies List

```
tensorflow==2.15.0
numpy==1.24.3
opencv-python==4.8.1.78
matplotlib==3.8.0
seaborn==0.13.0
scikit-learn==1.3.2
pandas==2.1.3
jupyterlab==4.0.9
ipykernel==6.26.0
tqdm==4.66.1
h5py==3.10.0
```

---

## 💻 Usage

### Training the Model

```bash
# Activate environment
conda activate task2

# Run main script
python main.py
```

**What happens during training:**
1. ✅ Automatically creates necessary directories
2. ✅ Downloads MNIST dataset (if not already cached)
3. ✅ Preprocesses data (normalization, reshaping)
4. ✅ Builds CNN model (225,034 parameters)
5. ✅ Trains for 10 epochs with validation
6. ✅ Evaluates on test set
7. ✅ Generates visualizations
8. ✅ Saves model and reports

**Expected Runtime:** ~4-5 minutes on CPU

### Loading Saved Model

```python
from tensorflow import keras

# Load the trained report
model = keras.models.load_model('models/simple_cnn_mnist.h5')

# Model is ready for predictions!
```

### Making Predictions

```python
import cv2
import numpy as np
from tensorflow import keras

# Load report
model = keras.models.load_model('models/simple_cnn_mnist.h5')

# Load and preprocess custom image
img = cv2.imread('your_digit.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=(0, -1))

# Predict
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)
confidence = prediction[0][predicted_digit] * 100

print(f"Predicted Digit: {predicted_digit}")
print(f"Confidence: {confidence:.2f}%")
```

---

## 🧠 Model Architecture

### Network Design

```
Input (28x28x1 grayscale image)
        ↓
Conv2D (32 filters, 3x3, ReLU)
        ↓
MaxPooling2D (2x2)
        ↓
Conv2D (64 filters, 3x3, ReLU)
        ↓
MaxPooling2D (2x2)
        ↓
Flatten (1600 units)
        ↓
Dense (128 units, ReLU)
        ↓
Dropout (0.5)
        ↓
Dense (10 units, Softmax)
        ↓
Output (10 classes: 0-9)
```

### Architecture Details

| Layer | Type | Output Shape | Parameters |
|-------|------|--------------|------------|
| Input | Input | (28, 28, 1) | 0 |
| conv2d | Conv2D | (26, 26, 32) | 320 |
| max_pooling2d | MaxPooling2D | (13, 13, 32) | 0 |
| conv2d_1 | Conv2D | (11, 11, 64) | 18,496 |
| max_pooling2d_1 | MaxPooling2D | (5, 5, 64) | 0 |
| flatten | Flatten | (1600) | 0 |
| dense | Dense | (128) | 204,928 |
| dropout | Dropout | (128) | 0 |
| dense_1 | Dense | (10) | 1,290 |

**Total Parameters:** 225,034  
**Trainable Parameters:** 225,034  
**Non-trainable Parameters:** 0  
**Model Size:** 879.04 KB

### Training Configuration

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy
- **Batch Size:** 128
- **Epochs:** 10
- **Validation Split:** 20%

---

## 📈 Performance Metrics

### Training Progress

| Epoch | Training Loss | Training Acc | Val Loss | Val Acc |
|-------|--------------|--------------|----------|---------|
| 1 | 0.3305 | 89.88% | 0.0799 | 97.68% |
| 2 | 0.1038 | 96.86% | 0.0552 | 98.40% |
| 3 | 0.0761 | 97.73% | 0.0437 | 98.73% |
| 4 | 0.0603 | 98.14% | 0.0436 | 98.71% |
| 5 | 0.0531 | 98.40% | 0.0396 | 98.95% |
| 6 | 0.0461 | 98.62% | 0.0373 | 98.98% |
| 7 | 0.0379 | 98.75% | 0.0439 | 98.83% |
| 8 | 0.0343 | 98.93% | 0.0370 | 99.02% |
| 9 | 0.0322 | 98.93% | 0.0373 | 99.01% |
| 10 | 0.0298 | 99.08% | 0.0339 | 99.08% |

### Key Observations

✅ **Fast Convergence:** Model reached 97%+ accuracy in first epoch  
✅ **No Overfitting:** Training and validation accuracy remain close  
✅ **Stable Training:** Loss decreases steadily without fluctuations  
✅ **Excellent Generalization:** Test accuracy (99.14%) exceeds validation (99.08%)  

---



## 🔍 Dataset Information

### MNIST Dataset

- **Source:** [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Training Samples:** 60,000 images
- **Test Samples:** 10,000 images
- **Image Size:** 28x28 pixels (grayscale)
- **Classes:** 10 (digits 0-9)
- **Format:** Normalized pixel values [0, 1]

### Data Preprocessing

1. **Normalization:** Pixel values divided by 255 (0-255 → 0-1)
2. **Reshaping:** Added channel dimension (28, 28) → (28, 28, 1)
3. **Train-Val Split:** 80% training, 20% validation
4. **No Augmentation:** Used original images without augmentation

---

## 🎓 Learning Outcomes

This project demonstrates understanding of:

✅ Convolutional Neural Networks (CNNs)  
✅ Image classification pipelines  
✅ Data preprocessing and normalization  
✅ Model training and validation  
✅ Performance evaluation metrics  
✅ Model serialization (saving/loading)  
✅ Visualization of results  
✅ Python programming and TensorFlow/Keras  

---

## 🔮 Future Enhancements

### Potential Improvements

- [ ] **Data Augmentation:** Add rotation, scaling, shifting for better generalization
- [ ] **Advanced Architecture:** Try ResNet, VGG, or MobileNet
- [ ] **EMNIST Dataset:** Extend to letters (A-Z) recognition




## 📚 References

1. [MNIST Database](http://yann.lecun.com/exdb/mnist/)
2. [TensorFlow Documentation](https://www.tensorflow.org/)
3. [Keras Documentation](https://keras.io/)
4. [CNN Architecture Guide](https://cs231n.github.io/)
5. [Deep Learning Book](https://www.deeplearningbook.org/)

---

## 📧 Contact

For questions, suggestions, or issues:
- https://github.com/aule-08
- Email: va7kmilindamadhushan@gmail.com

---

## 🎉 Project Status

**Status:** ✅ **COMPLETED**

**Final Results:**
- Test Accuracy: **99.14%**
- Training Time: ~4.5 minutes
- Model Size: 879 KB
- All objectives achieved successfully!

---

**Last Updated:** November 20, 2025

**Version:** 1.0.0


