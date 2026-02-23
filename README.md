# Capstone II

## Overview

This project investigates robust object detection under low-light conditions using the AI-Hub low-illumination dataset.

To address illumination variability and scene redundancy, we designed a scene-aware data balancing pipeline and a brightness-stage adaptive augmentation framework. Detection was performed using YOLOv11.

The primary objective was to improve detection robustness in extreme lighting environments through data-centric optimization rather than architectural modification.

---

## Dataset

- Source: AI-Hub Low-light Object Detection Dataset
- Brightness stages:
  - Stage 1: 24,888 images
  - Stage 3: 24,895 images
  - Stage 5: 24,867 images
- Total JSON scanned: 74,650
- Final usable images: ~73,634
- Final classes after rare-class filtering: ~28 classes

### Scene Grouping Strategy

- Scene groups created using filename-based temporal logic (±30 frames)
- Total scene groups: ~2,300
- Maximum 30 images per scene to prevent scene dominance
- Maximum 250 samples per class to reduce imbalance
- Train / Val / Test split: 70 / 15 / 15

---

## Data-Centric Engineering Pipeline

### 1️ Rare Class Filtering
- Removed classes with ≤300 instances
- Reduced noise and stabilized detector training

### 2️ Scene-aware Sampling
- Controlled temporal redundancy
- Prevented near-duplicate frames from biasing the model

---

## Brightness-Stage Adaptive Augmentation

### Step 1: Brightness Stage Classifier

A ResNet18-based supervised classifier was trained to predict brightness stage (stage1, stage3, stage5) directly from pixel data.

- Backbone: ResNet18
- Classes: 3
- Loss: CrossEntropy
- Optimizer: Adam
- Epochs: 10

This enables content-based illumination estimation without metadata dependency.

---

### Step 2: Stage-Specific Illumination Augmentation

After brightness prediction:

- Stage 1 → Gamma correction + contrast enhancement
- Stage 3 → Mild color perturbation
- Stage 5 → Controlled darkening

This avoids unrealistic augmentation artifacts and preserves illumination distribution.

---

### Step 3: YOLOv11 BBox-aware Augmentation

- Horizontal Flip
- Small-angle Rotation
- Resize (640×640)
- Bounding box visibility filtering

---

## Model Training

- Detector: YOLOv11
- Input size: 640×640
- 41 object categories
- Custom dataset.yaml configuration
- Data-centric preprocessing applied

---

## Key Contributions

- Scene-aware sampling strategy to control temporal redundancy
- Rare-class filtering to improve detection stability
- Brightness-stage classifier for content-aware preprocessing
- Stage-adaptive augmentation pipeline
- Data-centric enhancement of YOLOv11 performance under low-light conditions

---

## Experimental Results

We compared the baseline YOLOv11 model trained on the original balanced dataset with the proposed brightness-stage adaptive augmentation pipeline.

Unexpectedly, the baseline YOLOv11 achieved slightly higher mAP compared to the stage-adaptive augmentation model.

This suggests that:
- Excessive illumination manipulation may distort feature distribution
- YOLO’s built-in augmentation may already be sufficient
- Stage-specific augmentation requires more precise parameter tuning

These findings highlight the importance of validating data-centric approaches with controlled experiments.

---

## Repository Structure

- `data_processing/` : Scene grouping & balancing scripts
- `brightness_classifier/` : ResNet18 brightness prediction model
- `augmentation/` : Stage-aware augmentation pipeline
- `yolo_training/` : YOLOv11 configuration and training
