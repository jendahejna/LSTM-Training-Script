# LSTM Air Temperature Prediction from Microwave Link Data

# LSTM Air Temperature Prediction from Microwave Link Data

![Status](https://img.shields.io/badge/Status-Development-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## Description

This project implements a **Long Short-Term Memory (LSTM) neural network** for the task of **regression prediction of air temperature**. The model leverages data collected from **commercial microwave links**, providing an innovative approach to environmental temperature sensing. This script focuses on the training and evaluation pipeline for the LSTM model, including data loading, preprocessing, scaling, model definition, training with TensorFlow's `tf.data` API, and performance evaluation.

The goal is to accurately predict air temperature based on various features derived from microwave link measurements and auxiliary data, demonstrating the potential of machine learning in environmental monitoring.

---

## Key Features

* **Data Loading & Preprocessing**: Efficiently loads and preprocesses data from multiple CSV files, selecting relevant features and engineering new ones (e.g., Day of Year, Hour) from timestamps.
* **Data Scaling**: Utilizes `StandardScaler` from `scikit-learn` to standardize input features, which is crucial for neural network performance. The scaler is saved for later use in inference.
* **LSTM Model Architecture**: Implements a deep LSTM network with two LSTM layers (160 and 200 units) followed by a dense output layer.
* **TensorFlow `tf.data` Pipeline**: Leverages `tf.data.Dataset` for optimized and scalable data input during training, validation, and testing.
* **Training Optimization**: Incorporates `EarlyStopping` and `ReduceLROnPlateau` callbacks to prevent overfitting and optimize the learning rate dynamically.
* **Performance Evaluation**: Calculates and reports standard regression metrics: **Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score**.
* **Result Visualization**: Generates a scatter plot comparing actual vs. predicted temperatures, providing a visual assessment of model performance.
* **Model Persistence**: Saves the trained LSTM model and the fitted `StandardScaler` for future inference without retraining.
* **GPU Acceleration**: Configured to automatically detect and utilize available GPUs for faster training, if present.

---

## Technical Stack

* **Python**: The core programming language.
* **TensorFlow/Keras**: For building, training, and evaluating the LSTM neural network.
* **Pandas**: For efficient data loading, manipulation, and preprocessing.
* **NumPy**: For numerical operations and array manipulation.
* **Scikit-learn**: For data splitting and feature scaling.
* **Matplotlib**: For plotting and visualizing model results.
* **Joblib**: For saving and loading the `StandardScaler` object.

---

## File Structure
