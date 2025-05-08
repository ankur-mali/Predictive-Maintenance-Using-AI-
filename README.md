# AeroEngine Health Dashboard

An interactive dashboard for **predictive maintenance** of turbofan engines. This project leverages NASA's **CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset to monitor engine health, analyze sensor trends, and predict the Remaining Useful Life (RUL) of aero-engines. Built with [Streamlit](https://streamlit.io/), the dashboard provides real-time visual insights and predictive analytics to help optimize maintenance scheduling.
- Live At https://areoengine.streamlit.app/
---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Installation & Usage](#installation--usage)
- [License](#license)

---

## Overview

The AeroEngine Health Dashboard is designed to:

- **Monitor Engine Health:** Visualize the distribution of engine health metrics and identify engines that require maintenance.
- **Analyze Sensor Trends:** Track sensor data over operational cycles to detect trends and anomalies.
- **Predict Maintenance Needs:** Use a machine learning model (Random Forest Regressor) to estimate the Remaining Useful Life (RUL) of engines, allowing for proactive maintenance planning.

By processing the CMAPSS dataset, the dashboard transforms raw sensor readings and operational settings into actionable insights for predictive maintenance.

---
![image](https://github.com/user-attachments/assets/f48769aa-df03-4d78-9727-295dcb13015a)

## Features

- **Interactive Dashboard:** Built with Streamlit for an intuitive and user-friendly experience.
- **Data Preprocessing:**
  - Reads and cleans the raw training, testing, and RUL data.
  - Eliminates low-variance sensor features and normalizes data using `MinMaxScaler`.
  - Calculates the RUL for each training sample.
- **Predictive Modeling:**
  - Trains a `RandomForestRegressor` to predict the RUL.
  - Splits the data into training and validation sets to ensure robust performance.
- **Data Visualization:**
  - **Engine Health Overview:** Histogram displaying engine RUL distribution with a maintenance threshold indicator.
  - **Sensor Analytics:** Line plots depicting sensor trends over time.
  - **Maintenance Predictions:** Scatter plots comparing predicted versus actual RUL values along with performance metrics.
- **Downloadable Data:** Allows users to export sensor data as CSV files for further analysis.

---
![image](https://github.com/user-attachments/assets/a6c76d31-6459-440c-94ad-163c1a4d4aca)


## Technology Stack

- **Python** – Core programming language.
- **Streamlit** – Framework for building interactive dashboards.
- **Pandas & NumPy** – Data manipulation and numerical computations.
- **Matplotlib & Seaborn** – Data visualization.
- **scikit-learn** – Machine learning and data preprocessing.

---

## Dataset Information

This project uses NASA’s **CMAPSS dataset**, a benchmark in prognostics and health management for turbofan engines.

- **Description:** The dataset comprises multivariate time-series data including sensor readings, operational settings, and engine performance metrics.
- **Structure:**
  - **Training Data:** Historical engine performance data used to model degradation.
  - **Test Data:** Recent sensor readings to evaluate model predictions.
  - **True RUL:** Ground truth values representing the remaining cycles before maintenance is needed.
- **Application:** The dataset is ideal for developing models that predict when an engine will require maintenance, thus preventing unexpected failures and reducing downtime.

---

## Project Structure 
- ├── README.md # Project documentation (this file) -
- ├── app.py # Main Streamlit application code 
- ├── style.css # Custom CSS styling for the dashboard
- ├── train\_FD001.txt # Training dataset from CMAPSS
- ├── test\_FD001.txt # Test dataset from CMAPSS
- ├── RUL\_FD001.txt # Ground truth RUL values
- └── requirements.txt # Python dependencies


---

## Key Components

### Data Loading and Preprocessing

- **Function:** `load_data()`
- **Description:** Reads the training, testing, and RUL datasets, cleans the data by dropping low-variance sensor features, scales the features using `MinMaxScaler`, and computes the Remaining Useful Life (RUL) for the training samples.

### Model Training

- **Function:** `train_model()`
- **Description:** Splits the preprocessed data into training and validation sets, trains a `RandomForestRegressor` on the training data, and caches the trained model for quick loading during dashboard interactions.

### Interactive Visualizations

- **Engine Health Overview:** Displays a histogram of engine RUL values with an adjustable maintenance threshold.
- **Sensor Analytics:** Provides line plots for selected engine sensor data trends over operational cycles.
- **Maintenance Predictions:** Compares predicted RUL against actual RUL with performance metrics (MAE, accuracy) and a scatter plot for visual analysis.
- **Download Functionality:** Offers an option to download sensor data for a selected engine in CSV format.

---

## Installation & Usage

### Prerequisites

- Python 3.7 or higher

### Installation

1. **Clone the repository:**

   ```bash
   https://github.com/ankur-mali/Predictive-Maintenance-Using-AI-
   

