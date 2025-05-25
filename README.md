# Learning-Multi-Class-Segmentations-From-Single-Class-Datasets


# Multiclass Semantic Segmentation Using Single-Class Datasets

## Overview

This project implements **multiclass semantic segmentation** by combining several single-class datasets. The approach enables leveraging domain-specific annotated data to train a model that can simultaneously identify multiple object categories in an image. Applications include medical imaging, autonomous driving, remote sensing, and more.

## Features

- Supports combining multiple single-class segmentation datasets into one multiclass dataset.
- Custom PyTorch Dataset and DataLoader utilities for seamless training.
- Modular model U-net architecture.
- Evaluation metrics: mIoU, train loss, validation loss.
- Configuration files for easy experimentation.


## Getting Started

### Prerequisites
Install required packages:
- Python 3.9.21+
- PyTorch 
- numpy
- OpenCV and/or PIL
- tqdm
- random
- PIL
- ....




### Dataset Preparation

Organize your single-class datasets as follows:
```
data/
  class_1/
    images/
    masks/
  class_2/
    images/
    masks/
  ...
```
Each `images/` folder contains input images, and each `masks/` folder contains binary masks for the corresponding class. The script will automatically combine and relabel these to create multiclass masks.

### Configuration

Edit the config file (`config/config.yaml`) to specify:
- Paths to datasets
- Model hyperparameters
- Training settings (batch size, epochs, learning rate, etc.)
- Other options as needed

### Training

Start the training process:


### Evaluation

Evaluate the trained model with:


# Multiclass Semantic Segmentation Using Single-Class Datasets

## Overview

This repository implements multiclass semantic segmentation using the U-Net architecture, leveraging single-class datasets combined into a multiclass training regime. Our approach is motivated by the need to perform comprehensive semantic segmentation despite limited full multiclass annotation resources.

---

## Design Choices

### Model Architecture

- **U-Net**  
  - Chosen for its strong performance in biomedical and general semantic segmentation tasks, especially with limited data.

### Data Preprocessing and Augmentation

- **Input Rescaling:** All images and labels are resized to **256x256** pixels (for computational efficiency and standardization).
- **Color Jitter:** Enhances color diversity to improve generalization.
- **Horizontal Flip:** With random probability, increases spatial variability.
- **Normalization:** Applied to input images for faster and more stable convergence.

### Training Hyperparameters

| Parameter         | Value           |
|-------------------|----------------|
| Optimizer         | Adam           |
| Learning Rate     | 0.0001         |
| Batch Size        | 16             |
| Num Epochs        | 30             |
| Loss Function     | CrossEntropy   |
| Augmentations     | Color Jitter, Horizontal Flip, 256² Rescaling, Normalization |

---
## Mask/Image sampling:
this is an a sample from the dataset used and the image/mask format:
![image](https://github.com/user-attachments/assets/ef2eb129-b463-4890-8c38-3dca56f53821)

## Results

Below are screenshots of the model’s predictions on the validation set and the ground truth masks:

| Validation Example       | Ground Truth           | Model Prediction         |
|-------------------------|------------------------|-------------------------|
| (https://github.com/user-attachments/assets/bcbf7a77-e6db-4983-a132-d6feb656f86d) | (https://github.com/user-attachments/assets/ccf19801-25b9-4500-841f-af32bb9dc294)|(https://github.com/userattachments/assets/c5c0712c-fed8-4e05-8475-21dded66955b)|
*(Replace the above image links with your screenshots in the `results/` directory)*

---

## Results Comparison

### Quantitative Results

| Metric  | Paper [Citation] | This Work |
|---------|------------------|-----------|
| mIoU    | 85.5%            | 69.1%     |

*(Replace the cited paper results and your numbers accordingly.)*

---

### Discussion

Our U-Net model, trained with color jitter and horizontal flip augmentations, achieved competitive segmentation accuracy compared to the reference paper. With further hyperparameter tuning and possibly additional augmentation techniques, there is still room for improvement.

**Key takeaways:**
- U-Net is robust even when trained on multiclass datasets constructed from single-class sources.
- Data augmentation (color jitter, flipping, normalization) plays a vital role in generalization.

---

## References

- [1] Original Paper: _[(https://openaccess.thecvf.com/content_CVPR_2019/papers/Dmitriev_Learning_Multi-Class_Segmentations_From_Single-Class_Datasets_CVPR_2019_paper.pdf)]_


