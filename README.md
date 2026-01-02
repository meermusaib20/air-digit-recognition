Air Digit Recognition using PyTorch & MediaPipe

A real-time digit recognition project using PyTorch and MediaPipe, trained on the MNIST dataset and integrated with an air-drawing hand gesture interface for live digit prediction.

Project Overview

This project has two parts:

Model Training & Evaluation

A neural network is trained on the MNIST handwritten digit dataset.

Accuracy is evaluated on the official MNIST test set.

Real-Time Air-Drawing Demo

Uses MediaPipe hand tracking to detect the index finger.

Users draw digits in the air.

The trained model predicts the digit in real time.

Training and evaluation are kept separate from the demo, following correct machine learning practices.

Tech Stack

Python

PyTorch

MediaPipe

OpenCV

MNIST Dataset

Project Structure

train_and_evaluate.py – train model and compute accuracy
evaluate_saved_model.py – evaluate saved model
air_draw.py – real-time air-drawing demo
hand_landmarker.task – MediaPipe hand model

How to Run
Train and evaluate the model

python train_and_evaluate.py

Run the air-draw demo

python air_draw.py

Controls:

Draw using index finger

p → predict

c → clear

q → quit

Model Performance

Dataset: MNIST

Model: Fully connected neural network

Accuracy: ~96–97%

Accuracy is evaluated offline using labeled test data.

Author

Meer Musaib
Computer Science Student | AI & ML Enthusiast
