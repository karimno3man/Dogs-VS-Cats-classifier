# Dogs-VS-Cats-classifier
This project implements a Convolutional Neural Network (CNN) built entirely from scratch using TensorFlow and Keras, to classify images of dogs and cats. The model was trained and evaluated on the Kaggle "Dogs vs Cats" dataset, containing 25,000 labeled images (12,500 dogs and 12,500 cats). 

# 🐶🐱 Dogs vs Cats Image Classifier (Custom CNN)

## 🧠 Project Overview

This project implements a **Custom Convolutional Neural Network (CNN)** model for **binary image classification** — distinguishing between **cats 🐱** and **dogs 🐶** using the popular Kaggle *Dogs vs Cats* dataset.  

The model is built **from scratch (no transfer learning)** to demonstrate how convolutional layers, pooling, and dropout regularization can be used effectively to achieve strong results on a large-scale image dataset.

---

## 📂 Dataset

**Dataset:** [Dogs vs Cats – Kaggle](https://www.kaggle.com/datasets/biaiscience/dogs-vs-cats)  
**Total Images:** 25,000  
- 🐱 **Cats:** 12,500  
- 🐶 **Dogs:** 12,500  

The dataset was split into:
- **70% Training**
- **15% Validation**
- **15% Testing**

Each image was resized to **224×224 pixels** and normalized to `[0, 1]` for model input.

---

## 🏗️ Model Architecture

The CNN was designed from scratch using **TensorFlow/Keras**, with an emphasis on depth, normalization, and dropout for generalization.

```python
Model = Sequential([
    Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(256, (3, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),

    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.6),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),

    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
```

Optimizer: Adam (lr=0.001)
Loss Function: Binary Crossentropy
Metrics: Accuracy


⚙️ Training Details

Epochs: 60

Batch Size: 32

Steps per Epoch: 625

Validation Steps: 156

Hardware: GPU (TensorFlow with CUDA)

| Metric       | Training | Validation |
| :----------- | :------: | :--------: |
| **Loss**     |  0.0325  |   0.2784   |
| **Accuracy** |  98.92%  |   90.95%   |


📈 Results Interpretation

The custom CNN achieved ~99% training accuracy and ~91% validation accuracy, showing strong generalization and effective feature learning.

Validation loss is slightly higher than training, indicating mild overfitting, but performance remains excellent for a custom-built model.

The progressive convolutional depth, combined with Batch Normalization and Dropout, helped prevent gradient vanishing and improved convergence.

Relu activations were chosen for non-linearity, while sigmoid output handled binary class prediction.

Adam optimizer enabled stable and efficient learning without manual tuning of momentum or decay.

<img width="1589" height="590" alt="image" src="https://github.com/user-attachments/assets/57427cb6-4c3f-4661-b971-16f198b7110e" />

