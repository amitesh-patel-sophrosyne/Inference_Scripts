# Inference Repository

This repository contains the inference pipelines for all the machine learning models and algorithms currently used in the project.

## Available Inference Pipelines

The repository currently includes inference scripts for:

* Arrhythmia Models
* IMU Models
* PPG Algorithm
* ECG Parameter Extraction

## Model Files

The latest trained model files are **not included** in this repository.

Download the latest model directory from the link below and place it in the appropriate location before running inference.

**Model Download:**
`<.....>`

## Test Data

Sample test data is also available for validating the inference pipelines.

**Test Data Download:**
`<....>`

## Purpose of This Repository

This repository serves as the **reference implementation** for all model inference. It provides the core inference logic that can be reused across different deployment environments.

The deployment implementation may vary depending on the target architecture, but the inference logic should remain consistent.

Examples include:

* Combining multiple IMU models into a single inference pipeline that returns all IMU-related predictions in one request.
* Extending the fall detection model with additional post-fall analysis.
* Running preprocessing steps such as filtering, normalization, motion artefact removal, and signal quality assessment before inference.
* Handling different input sources (mobile devices, APIs, message queues, streaming data, etc.).
* Managing data validation, batching, logging, and result formatting based on deployment requirements.

Instead of rewriting model inference, deployment services should import and build upon the inference modules provided in this repository.

In short:

* **This repository:** Core model inference and preprocessing logic.
* **Deployment repository:** Data ingestion, orchestration, API integration, infrastructure-specific processing, and result delivery.

This separation keeps the inference logic modular, reusable, easier to test, and consistent across all deployment architectures.
