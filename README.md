# Emotion Detection

## Overview

This project implements an **Emotion Detection System** capable of recognizing **7 different emotions** from facial images. It supports both **image uploads** and **real-time webcam detection**, using deep learning and computer vision techniques.

The system is deployed with **FastAPI** for easy API access and can be tested locally or integrated into other applications.

---

## Features

* Detects 7 emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.
* Preprocessing with **MediaPipe** for accurate face detection.
* **Data Augmentation** applied to balance classes and improve generalization.
* Uses **Transfer Learning** with **MobileNet** and **ResNet**.
* **FastAPI** backend to serve predictions.
* Evaluate performance using **confusion matrix** and accuracy metrics.

---

## Technologies Used

* **Python 3.x**
* **TensorFlow / Keras** – for model training
* **MediaPipe** – for face detection
* **OpenCV** – for image processing and real-time camera input
* **FastAPI** – for API deployment
* **Matplotlib / Seaborn** – for visualization
* **Conda** – environment management

---

## Project Workflow

1. **Dataset Loading:** Loaded dataset and inspected sample images.
2. **Data Augmentation:** Applied augmentations to balance classes and increase the number of images.
3. **Face Detection:** Used **MediaPipe** to detect faces before passing them to the model.
4. **Model Training:**

   * Used **MobileNet** and **ResNet** with transfer learning.
   * Replaced original classifiers with a new dense layer for 7 emotion classes.
   * MobileNet achieved **68.5% accuracy**, ResNet achieved **70% accuracy**.
5. **Evaluation:**

   * Confusion matrix used to evaluate per-emotion prediction performance.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/emotion-detection.git
   cd emotion-detection
   ```
2. Create and activate conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate face-reg
   ```
3. Install any remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

### 1. Using FastAPI

```bash
uvicorn main:app --reload
```

* The API will be available at `http://127.0.0.1:8000`.
* Use `/predict` endpoint to upload an image and get the predicted emotion.

### 2. Using Jupyter Notebook

```bash
jupyter notebook
```

* Explore training, evaluation, and visualizations interactively.

---

## Results

* **MobileNet Accuracy:** 68.5%
* **ResNet Accuracy:** 70%
* Confusion matrix highlights which emotions were predicted more accurately.

---

## Future Improvements

* Add **multi-face detection** in real-time video.
* Deploy as a **web application** with live webcam emotion detection.
* Implement **emotion trend analysis** over time.


