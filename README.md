# Pediatric Bone Age Prediction from Hand Radiographs

**Deep-learning regression of skeletal age from X-ray images: a tuned custom CNN, VGG16 transfer learning, and ensemble methods**

**Author:** Prachi · MSc Computer Science, University of Delhi

---

## Overview

Bone age assessment is a routine paediatric radiology task. A clinician compares a child's left-hand X-ray against reference atlases to estimate skeletal maturity. A large gap between bone age and actual age can signal growth disorders or endocrine and genetic conditions. Manual assessment is slow and varies between readers, which makes it a natural target for automated image regression.

This project predicts **bone age in months** directly from hand radiographs and compares four approaches:

| # | Model | Idea |
|---|---|---|
| 1 | **Custom CNN** | Convolutional network trained from scratch; architecture and learning rate chosen by hyperparameter search |
| 2 | **VGG16 transfer learning** | ImageNet-pretrained VGG16 backbone (frozen) with a tuned regression head |
| 3 | **Weighted ensemble** | Error-weighted average of the CNN and VGG16 predictions |
| 4 | **Stacking ensemble** | Linear-regression meta-model trained on the two models' validation-set predictions |

**Best result:** the stacking ensemble reached **MAE = 16.66 months** and **R² = 0.723** on a held-out test set of 2,523 radiographs.

---

## Dataset

**RSNA Pediatric Bone Age Challenge (2017)**, a public dataset of left-hand radiographs released by the Radiological Society of North America.

| Property | Value |
|---|---|
| Images used | 12,611 hand radiographs (PNG) |
| Target | Bone age in months (`boneage` column) |
| Age range | ~1 to 228 months (0–19 years) |
| Metadata | Image ID, bone age, sex (sex not used as an input in this version) |

---

## Methodology

### 1. Preprocessing
- Each radiograph is resized to **224 × 224 × 3** and pixel values are scaled to **[0, 1]**.
- Missing or unreadable image files are detected and excluded, and image–label alignment is checked before training.

### 2. Data split

| Set | Share | Images |
|---|---|---|
| Training | 64% | 8,070 |
| Validation | 16% | 2,018 |
| Test (held out) | 20% | 2,523 |

The test set is never used for training or hyperparameter selection. Splits are fixed with `random_state=42`.

### 3. Custom CNN: hyperparameter search with Keras Tuner

A random search (10 trials, up to 20 epochs each, early stopping on validation loss with patience 3) over:

| Hyperparameter | Search space | Best value |
|---|---|---|
| Conv block 1 filters | 32 – 64 | **64** |
| Conv block 2 filters | 64 – 128 | **96** |
| Extra (third) conv block | yes / no | **yes** (filters searched in 128 – 256) |
| Dense units | 256 – 512 | **384** |
| Dropout | 0.1 – 0.5 | **0.3** |
| Learning rate (Adam) | 1e-3, 1e-4, 5e-5 | **1e-3** |

Architecture: stacked Conv2D (3×3, ReLU) + MaxPooling blocks → Flatten → Dense (ReLU) → Dropout → Dense(1, linear). Trained with mean-squared-error loss. Search time was about 9 hours on CPU.

### 4. VGG16 transfer learning

The **VGG16 convolutional base pretrained on ImageNet** is used as a frozen feature extractor, with a trainable regression head (Flatten → Dense → Dropout → Dense(1)). Random search: 5 trials, up to 15 epochs each.

| Hyperparameter | Search space | Best value |
|---|---|---|
| Dense units | 256 – 512 | **512** |
| Dropout | 0.1 – 0.5 | **0.4** |
| Learning rate (Adam) | 1e-3, 1e-4, 5e-5 | **1e-4** |

### 5. Ensembles

- **Weighted ensemble:** `ŷ = α · ŷ_CNN + (1 − α) · ŷ_VGG16`, with `α = MSE_VGG / (MSE_CNN + MSE_VGG)` so that the model with lower error gets more weight (α = 0.405).
- **Stacking ensemble:** a `LinearRegression` meta-model learns how to combine the two models' predictions. It is fit on the **validation-set** predictions, so the test set stays unseen, then applied to the test-set predictions.

---

## Results (held-out test set, n = 2,523)

| Model | MAE (months) ↓ | RMSE (months) ↓ | MSE ↓ | R² ↑ |
|---|:---:|:---:|:---:|:---:|
| Custom CNN (tuned) | 24.44 | 31.46 | 989.49 | 0.441 |
| VGG16 (tuned) | 17.50 | 22.81 | 520.29 | 0.706 |
| Weighted ensemble | 20.06 | 25.92 | 671.81 | 0.621 |
| **Stacking ensemble** | **16.66** | **22.14** | **490.05** | **0.723** |

<p align="center">
  <img src="images/actual_vs_predicted.jpg" width="720" alt="Actual vs predicted bone age for the stacking ensemble on the test set">
</p>

<p align="center">
  <img src="images/mae_comparison.jpg" width="600" alt="Mean absolute error comparison across the four models">
</p>

### Key findings

- **Transfer learning clearly beats training from scratch.** The frozen VGG16 features reduced MAE by about 7 months compared with the custom CNN. ImageNet features carry over well to radiograph texture and shape, even without fine-tuning.
- **Stacking is the best combiner.** Learning the combination weights on validation data improved on VGG16 alone (MAE 17.50 → 16.66). Simple error-weighted averaging did worse than VGG16 alone, because the weaker CNN pulled the average away from the better model.
- **Predictions are pulled towards the middle of the age range.** The scatter plot shows that the youngest children tend to be over-estimated and the oldest under-estimated. This is a common pattern for MSE-trained regressors on skewed age distributions.
- **Evaluation note:** the weight α in the weighted ensemble was computed from test-set errors, so that row is slightly optimistic. The stacking ensemble, which is the headline result, uses only validation data to learn its weights.

---

## Repository contents

| File | Description |
|---|---|
| `bone_age_prediction.ipynb` | Complete pipeline: data loading, preprocessing, CNN and VGG16 hyperparameter search, ensembling, evaluation and plots |
| `images/` | Result figures used in this README |
| `README.md` | This document |

---

## Running the notebook

1. Download the RSNA Bone Age dataset from Kaggle and place `boneage-training-dataset.csv` and the `boneage-training-dataset/boneage-training-dataset/` image folder next to the notebook.
2. Install the dependencies:
   ```bash
   pip install tensorflow keras-tuner scikit-learn pandas numpy matplotlib pillow
   ```
3. Run the notebook top to bottom. The full hyperparameter search takes about 19 hours on CPU (about 9 h for the CNN, 10 h for VGG16), and a GPU is strongly recommended. Keras Tuner saves its progress in `keras_tuner_cnn/` and `keras_tuner_vgg/`, so an interrupted search resumes where it stopped.

---

## Tech stack

Python · TensorFlow / Keras · Keras Tuner · scikit-learn · NumPy · pandas · Matplotlib

---

## References

1. Halabi, S. S., et al. (2019). The RSNA Pediatric Bone Age Machine Learning Challenge. *Radiology*, 290(2), 498–503.
2. Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*.
3. O'Malley, T., et al. (2019). Keras Tuner. https://github.com/keras-team/keras-tuner
