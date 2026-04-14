# Automatic-License-Plate-Recognition-System

Project Overview:
This project implements a modular Automatic License Plate Recognition (ALPR) system designed to detect and recognize vehicle license plates from video streams. Developed by a team of four, the project is a multi-stage computer vision system designed to automate the detection and recognition of vehicle license plates from video streams. The project specifically addresses real-world challenges like low-light environments and motion blur by integrating digital image processing with deep learning.

Features:
- Intelligent Frame Extraction: Uses Laplacian Variance to identify and skip blurry frames, ensuring only high-quality data enters the pipeline.
- Image Enhancement: Employs CLAHE (Contrast Limited Adaptive Histogram Equalization) and 4-bit quantization to normalize contrast and reduce noise in difficult lighting.
- Deep Learning Detection: Features a fine-tuned YOLOv8 model to localize plates with high confidence (0.78+).
- Custom OCR Engine: Utilizes Tesseract OCR with a character whitelist for accurate alphanumeric extraction from processed plate crops.
