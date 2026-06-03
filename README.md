# Traffic Accident Detection System

## Overview

This repository contains the source code and evaluation reports for a vision-based Traffic Accident Detection system. It utilises spatiotemporal analysis (Optical Flow) and deep learning to classify routine traffic flow versus catastrophic kinetic anomalies.

## Repository Structure

The project is systematically organised into the following key directories:

- **`annotations/`**: Contains data labelling records, including `accidents.csv`, `accidents_cleaned.csv`, `videos.csv`, and annotation progress tracking via `progress.json`.
- **`reports/`**: Stores generated evaluation metrics such as the `final_evaluation_report.xlsx`, confusion matrices, and performance curves (F1, Precision-Recall, and ROC).
- **`scripts/`**: The core execution pipeline, containing files for data generation, training, reporting, and inference.
- **`requirements.txt`**: The list of Python dependencies required to run the environment.

## Installation

To set up the environment, install the necessary Python dependencies using the provided requirements file:

    pip install -r requirements.txt

## How to Use

The workflow is divided into discrete scripts located in the `scripts/` directory. Follow this pipeline to prepare data, train the model, and deploy it:

### 1. Data Annotation and Preparation

If you are utilising custom video data, begin by annotating and cleaning your dataset.

- Use the modules inside `scripts/annotationScripts/` (such as `annotate_temporal_resume.py` and `data_cleaning.py`) to explicitly isolate accident frames from normal traffic.
- Filter your annotated datasets using `data_filter.py`.
- Split your raw data into training, validation, and testing sets using `scripts/split_data.py`.

### 2. Spatiotemporal Feature Extraction

Transform raw RGB video frames into dense optical flow tensors to capture unstructured kinetic energy.

- Run `scripts/generate_optical_flow.py` to extract spatiotemporal features from your primary dataset.
- To augment your data with a steady-state environmental baseline, use `scripts/gen_optical_flows_tad.py`.

### 3. Model Training

Once the optical flow data is balanced and prepared, initiate the deep learning training process.

- Execute `scripts/train_final.py` to train the classification architecture.

### 4. Evaluation and Reporting

Evaluate the statistical reliability and safety thresholds of your trained model.

- Run `scripts/generate_final_report.py`.
- This script will automatically generate visual plots (e.g., `F1_curve.png`, `roc_curve.png`, `confusion_matrix.png`) and tabular data (`final_evaluation_report.xlsx`) inside the `reports/` directory.

### 5. Real-Time Inference

To test the model on new, unseen surveillance footage with a live User Interface:

- Execute `scripts/inference_video.py`.

## Utility Scripts

The `scripts/UtilScripts/` folder contains supplementary helper modules:

- `extract_frames.py`: For extracting individual RGB frames from continuous video files.
- `optical_flow_test.py`: To debug and experimentally verify the optical flow extraction logic.
- `uploadToRobo.py`: For uploading datasets, annotations, or weights to external cloud storage.
