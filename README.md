# 🧠 CNN Model Implementation (AlexNet, GoogleNet, ResNet, VGG19) 🎯🚀

This project demonstrates the implementation of four popular **🌀 Convolutional Neural Networks (CNNs)** using both **🔥 PyTorch** and **💡 TensorFlow**. The models implemented are:-

- **📷 AlexNet**
- **🌍 GoogleNet (InceptionNet)**
- **🔄 ResNet (Residual Network)**
- **🖼 VGG19 (Visual Geometry Group 19-layer model)**

## 📌 Features 🛠✨
- Implemented in **🟠 PyTorch** and **🔵 TensorFlow**.
- 📦 Pre-trained weights support for transfer learning.
- 📊 Model training and evaluation on custom datasets.
- 📈 Performance visualization with metrics like accuracy and losses

## 📋 Prerequisites 🔧
Ensure you have the following installed

- 🐍 Python 3.8+
- 🔥 PyTorch
- 💡 TensorFlow
- 📷 OpenCV
- 📊 Matplotlib
- 🔢 NumPy
- 🖼 Torchvision (for loading pre-trained models)
- 🗂 TensorFlow Datasets (for loading sample datasets)

Install dependencies using:
```bash
pip install torch torchvision tensorflow opencv-python matplotlib numpy tensorflow-datasets
```

## 🚀 Model Implementations 🤖

### **📷 AlexNet**
- A deep CNN designed for large-scale **image classification**.
- First introduced in **🏆 ImageNet Classification (2012)**.

**🔥 PyTorch Implementation:**
```python
import torch
import torchvision.models as models

alexnet = models.alexnet(pretrained=True)
print(alexnet)
```

**💡 TensorFlow Implementation:**
```python
import tensorflow as tf
from tensorflow.keras.applications import AlexNet

alexnet = AlexNet(weights='imagenet')
alexnet.summary()
```

---

### **🌍 GoogleNet (InceptionNet)**
- Introduced in **📅 2014** (**Inception v1**).
- Uses **multiple kernel sizes** in a single layer.

**🔥 PyTorch Implementation:**
```python
googlenet = models.googlenet(pretrained=True)
print(googlenet)
```

**💡 TensorFlow Implementation:**
```python
googlenet = tf.keras.applications.InceptionV3(weights='imagenet')
googlenet.summary()
```

---

### **🔄 ResNet**
- Introduced in **📅 2015**.
- Uses **residual connections** to prevent vanishing gradients.

**🔥 PyTorch Implementation:**
```python
resnet = models.resnet50(pretrained=True)
print(resnet)
```

**💡 TensorFlow Implementation:**
```python
resnet = tf.keras.applications.ResNet50(weights='imagenet')
resnet.summary()
```

---

### **🖼 VGG19**
- Deep CNN with **🧩 19 layers**.
- Known for its **🔍 simplicity and depth**.

**🔥 PyTorch Implementation:**
```python
vgg19 = models.vgg19(pretrained=True)
print(vgg19)
```

**💡 TensorFlow Implementation:**
```python
vgg19 = tf.keras.applications.VGG19(weights='imagenet')
vgg19.summary()
```

## 📊 Training and Evaluation 🏋️‍♂️
### **🔥 Training (PyTorch)**
```python
import torch.optim as optim
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
```

### **💡 Training (TensorFlow)**
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=val_data)
```

## 📌 Notes 📝💡
- ✏️ Modify the **training** and **evaluation** scripts as per your dataset.
- ⚡ Use **GPU acceleration** (`CUDA` or `TensorFlow-GPU`) for **faster training**.
- 🛠 Experiment with **hyperparameters** to optimize performance.

## 📜 License 📄
This project is **open-source**. Feel free to modify and use it as needed! 💻🔓

---
💡 **Contributions and feedback are welcome!** 🚀
