# Drowsiness Detection System

## Overview
This project implements a **Drowsiness Detection System** to monitor a person's alertness while driving or working by detecting prolonged eye closure using a webcam or video feed. It utilizes a pre-trained TensorFlow model and OpenCV's Haar Cascade for real-time eye detection and classification.

---

## Features
- **Real-time eye detection** using Haar Cascade.
- **Drowsiness prediction** based on eye closure using a TensorFlow-based deep learning model.
- **Dynamic frame skipping** for optimized processing.
- Displays "Drowsy" warning on the screen when prolonged eye closure is detected.

---

## Requirements

### Hardware
- Webcam or a video feed device.

### Software
- Python 3.x
- Libraries:
  - keras
  - matplotlib
  - numpy
  - opencv
  - pandas
  - pillow
  - scipy
  - seaborn
  - tensorflow
  - torch
  - torchvision
  - ultralytics

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AnoopCA/Drowsiness-Detection-System.git
   cd drowsiness-detection-system