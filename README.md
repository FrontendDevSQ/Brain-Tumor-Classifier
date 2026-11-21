# Brain Tumor Classifier

A convolutional neural network (CNN) built with **PyTorch** that classifies brain MRI scans into **4 categories**. Trained and evaluated on Google Colab using a GPU.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for real diagnosis.

---

## Overview

Reviewing brain MRI scans by hand is slow and depends on specialist availability. This project explores how deep learning can act as an automated first-pass classifier to support faster screening.

The model takes an MRI image as input and predicts its class (for example, **glioma**). It is trained on a labeled training set and evaluated on a separate test set for loss and accuracy.

## Dataset

- Dataset is taken from Kaggle.
- Classes (4): glioma, meningioma, pituitary, no tumor

## Preprocessing

- Resize images to **128 x 128**
- Convert to tensors
- Normalize with mean `(0.5, 0.5, 0.5)` and std `(0.5, 0.5, 0.5)`

## Model Architecture

| Layer | Details |
|---|---|
| Conv Block 1 | Conv2d (3 → 32, kernel 3, padding 1) + ReLU + MaxPool (2) |
| Conv Block 2 | Conv2d (32 → 64, kernel 3, padding 1) + ReLU + MaxPool (2) |
| Conv Block 3 | Conv2d (64 → 128, kernel 3, padding 1) + ReLU + MaxPool (2) |
| Flatten | 128 x 16 x 16 features |
| Fully connected | Linear (32768 → 256) + ReLU + Dropout (0.5) |
| Output | Linear (256 → 4) |

## Training Setup

| Setting | Value |
|---|---|
| Framework | PyTorch, torchvision |
| Loss function | CrossEntropyLoss |
| Optimizer | AdamW (learning rate 1e-5) |
| Batch size | 32 |
| Epochs | 25 |
| Hardware | Google Colab, Tesla T4 GPU |

Training loss decreased steadily from **218.09** (epoch 1) to **67.17** (epoch 25), measured as total loss summed over batches.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **84.66819221967964 %** |
| Test loss | 0.36152733866946557  |

## How to Run

1. Open the notebook (`Brain_Tumor_Classifier.ipynb`) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Upload the dataset to your Google Drive and update the paths in the notebook:

```python
train_path = '/content/drive/MyDrive/Dataset/Training'
test_path  = '/content/drive/MyDrive/Dataset/Testing'
```

4. Choose **Runtime → Run all**. Allow Google Drive access when asked.
5. The last cells print the test loss, test accuracy, and show a random test image with its predicted and true class.

## Requirements

```
torch
torchvision
matplotlib
```

(All are pre-installed on Google Colab.)

## Future Improvements

- Add data augmentation (rotation, flips) to improve generalization
- Try transfer learning with pretrained models such as ResNet
- Report per-class precision, recall, and a confusion matrix
- Add a validation split and early stopping
- Build a simple web interface for uploading an MRI image

