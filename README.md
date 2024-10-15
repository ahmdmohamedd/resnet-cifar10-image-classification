# ResNet CIFAR-10 Image Classification

## Overview

This project utilizes transfer learning with the ResNet architecture to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The model is trained to identify and classify images into one of these ten categories.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Results](#results)

## Features

- Transfer learning with ResNet for efficient image classification
- Data preprocessing including normalization and augmentation
- Model training with adjustable hyperparameters
- Evaluation metrics including accuracy and loss

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- CIFAR-10 dataset

## Installation

To get started, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/resnet-cifar10-image-classification.git
   cd resnet-cifar10-image-classification
   ```

2. Create a virtual environment and activate it:
   ```bash
   conda create -n resnet-cifar10 python=3.8
   conda activate resnet-cifar10
   ```

3. Install the required packages:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

4. Download the CIFAR-10 dataset and place it in the `cifar-10-batches` directory:
   - You can download it from [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook resnet_cifar10_image_classification.ipynb
   ```

2. Run the cells in the notebook sequentially to preprocess the data, train the model, and evaluate its performance.

## Training the Model

The model is trained using the following key components:

- **Data Loaders**: Efficiently load and preprocess the CIFAR-10 images.
- **Model Architecture**: Utilize a pre-trained ResNet model and fine-tune it for the CIFAR-10 classification task.
- **Loss Function**: Use Cross Entropy Loss to measure model performance.
- **Optimizer**: Adam optimizer is employed for optimizing the model parameters.

## Results

After training the model, you can expect the following results:

- Training and validation loss over epochs
- Accuracy metrics for both training and validation datasets

Feel free to visualize the results using Matplotlib.
