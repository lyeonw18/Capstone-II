# Capstone II
Brightness-Stage Adaptive Augmentation for Robust Low-Light Object Detection

## 1. Overview
Conventional low-light enhancement research (e.g., Retinex-based deep models, GAN-based relighting) focuses on restoring visually pleasing images using complex restoration networks.

However:
- These methods optimize perceptual image quality rather than detection performance.
- They require an additional enhancement network, increasing inference latency and system complexity.
- Improvement in image quality does not guarantee improved object detection mAP.
Instead of restoring images, this study proposes a **data-centric brightness-stage adaptive augmentation framework**, allowing a YOLO-based detector to directly learn illumination variations without introducing extra networks.

The goal is to improve mAP under low-light conditions without architectural modification.

---

## 2. Dataset Construction
### Source
AI-Hub Low-light Object Detection Dataset
Due to large dataset size, only Brightness stages:
  - Stage 1: 24,888 images
  - Stage 3: 24,895 images
  - Stage 5: 24,867 images
- Total JSON scanned: 74,650
- Final usable images: ~73,634
- Final classes after rare-class filtering: ~28 classes

## 3. Class Balancing Strategy
- Scanned 74,650 labeled JSON annotations
- Total scene groups: ~2,300
- Removed classes with ≤300 instances
- Removed near-duplicate frames
- Selected 250 images per class
- Train / Val / Test split = 70 / 15 / 15

This significantly reduced class imbalance and improved stability.


## 4. Brightness Stage Classification Model
To avoid relying on filename metadata ("A/C/E"), we trained a content-based brightness classifier.
 - Backbone: ResNet18
 - Classes: Stage1 / Stage3 / Stage5
 - Loss: CrossEntropy
 - Optimizer: Adam
 - Epochs: 10
This allows illumination-aware augmentation directly from image content.

## 5. Augmentation Strategy
 ### Basic Augmentation
 - Horizontal Flip
 - ±10° Rotation
 - Brightness & Contrast shift
 - Color jitter
 - YOLO-format bbox transformation
 - Bboxes clipped if out-of-bound
 - Only boxes ≥20% visible retained

### Brightness-Stage Adaptive Gamma Augmentation
Stage 1:
   - Gamma = 1.05, 1.12
   - Enhance random.uniform(1.00–1.08)
Stage 3:
   - Gamma = 0.98, 1.02
   - Enhance random.uniform(0.95–1.05)
Stage 5:
   - Gamma = 0.92, 0.98
   - Enhance random.uniform(0.92–1.02)

Gamma ranges were redesigned per brightness stage to simulate realistic illumination variation while preserving scene consistency.


## 6. Model Training
- Detector: YOLOv11n
- Epochs: 50
- Batch size: 16
- Image size: 640×640
Two models were trained:
1. Basic augmentation only
2. Brightness-stage adaptive augmentation


## 7. Experimental Results
| Metric | Basic Aug | Stage-Adaptive Aug | Improvement |
|--------|------------|-------------------|-------------|
| Precision | 0.9310 | **0.9625** | +0.0315 |
| Recall | 0.8648 | **0.9652** | +0.1004 |
| mAP50 | 0.8788 | **0.9854** | +0.1066 |
| mAP50-95 | 0.8140 | **0.8303** | +0.0163 |

### Key Findings
- Significant improvement in mAP50 (+10.6%)
- Recall dramatically increased (+10%)
- High-confidence predictions reached ~0.99 precision
- Stable detection under unseen low-light generalization set (152 images)
Performance gain is attributed to:
  - Rare-class filtering
  - Balanced sampling
  - Stage-specific gamma redesign



## 8. Generalization Test
A separate set of 152 indoor/outdoor low-light images was collected.

The model successfully detected objects under diverse illumination conditions, confirming robustness beyond training distribution.


## 9. Conclusion
Brightness-stage adaptive augmentation:
  - Improves detection robustness without adding inference complexity
  - Outperforms baseline augmentation in key metrics
  - Provides a lightweight data-centric alternative to low-light restoration networks
This demonstrates that illumination modeling at the data level is highly effective for low-light object detection.


## 10. Repository Structure

- `data_processing/` : Scene grouping & balancing scripts
- `brightness_classifier/` : ResNet18 brightness prediction model
- `augmentation/` : Stage-aware augmentation pipeline
- `yolo_training/` : YOLOv11 configuration and training
