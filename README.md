# CIFAR-10 Image Classification
This notebook demonstrates an image classification task on the CIFAR-10 dataset. It explores two approaches:
1.  A simple feed-forward neural network.
2.  Transfer learning using a pre-trained ResNet50 model.

## Setup and Installation

To run this notebook, you'll need the following libraries. They are installed at the beginning of the notebook:

*   `kaggle`: For downloading the dataset.
*   `py7zr`: For extracting the dataset archives.
*   `tensorflow`: For building and training neural networks.
*   `numpy`, `pandas`, `matplotlib`, `seaborn`, `Pillow`, `opencv-python`: For data manipulation, visualization, and image processing.

  ## Data Preprocessing

1.  The `cifar-10.zip` file is extracted, followed by `train.7z` to access the image files.
2.  Labels are loaded from `trainLabels.csv` and mapped to numerical values (0-9).
3.  Image files (PNG format) are loaded, converted to NumPy arrays, and stored.
4.  The dataset is split into training and testing sets (80% train, 20% test).
5.  Image pixel values are scaled from `[0, 255]` to `[0, 1]`.

## Model Architectures

### 1. Simple Neural Network (for comparison)

A basic sequential model with:
*   `Flatten` layer to convert 32x32x3 images into a 1D array.
*   A `Dense` hidden layer with 64 units and `relu` activation.
*   A `Dense` output layer with 10 units (for 10 classes) and `softmax` activation.

This model serves as a baseline to demonstrate the improvement with more complex architectures like CNNs.

### 2. ResNet50 Transfer Learning

This model leverages the power of a pre-trained ResNet50 convolutional base. The architecture consists of:
*   `UpSampling2D` layers to increase image resolution to 256x256x3 (required by ResNet50).
*   `ResNet50` pre-trained on `imagenet` weights, with `include_top=False` to use it as a feature extractor.
*   `Flatten` layer.
*   `BatchNormalization` layers for stable training.
*   Two `Dense` hidden layers (128 and 64 units) with `relu` activation.
*   `Dropout` layers (0.5) for regularization.
*   A final `Dense` output layer with 10 units and `softmax` activation.

Both models are compiled using the `adam` or `RMSprop` optimizer, `sparse_categorical_crossentropy` loss, and `accuracy` or `acc` metrics.




## Training

The models are trained for 10 epochs with a validation split of 10% from the training data. The simple neural network achieved a validation accuracy of around 37%, while the ResNet50 transfer learning model achieved significantly higher validation accuracy (around 94%) after 10 epochs.
