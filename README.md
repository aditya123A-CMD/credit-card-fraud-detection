# credit-card-fraud-detection
# Credit Card Fraud Detection (ML-Based)

**Credit Card Fraud Detection** is a machine learning–powered system designed to identify potentially fraudulent credit card transactions in real time. The project combines data analysis, model training, and a web-based interface to demonstrate how ML models can be used to detect suspicious financial activity with high accuracy.

The system analyzes transaction attributes such as transaction amount, time, and engineered features to determine whether a transaction is legitimate or fraudulent. Using a trained machine learning model and preprocessing techniques, the application predicts fraud probability and returns instant results through a user-friendly dashboard.

## Key Features

* **Machine Learning–based Fraud Detection** – Detects fraudulent transactions using a trained classification model.
* **Real-Time Prediction** – Users can input transaction details and instantly receive fraud detection results.
* **Interactive Web Interface** – Built with Flask and HTML to simulate a real-time fraud monitoring dashboard.
* **High Performance** – Optimized prediction pipeline with fast response time.
* **Data Preprocessing & Feature Scaling** – Uses a trained scaler to standardize transaction inputs before prediction.

## Project Structure

* `data_generator.py` – Generates or prepares synthetic transaction data.
* `eda.py` – Performs exploratory data analysis to understand transaction patterns.
* `train_model.py` – Trains the machine learning model for fraud classification.
* `evaluate.py` – Evaluates model performance using metrics such as accuracy.
* `app.py` – Flask backend that serves the prediction API and web interface.
* `templates/index.html` – Frontend dashboard for transaction input and fraud detection results.
* `models/` – Contains trained model files, feature configuration, and scaler.
* `outputs/` – Stores datasets, evaluation results, and visualizations.

## Technologies Used

* **Python**
* **Scikit-learn**
* **Pandas & NumPy**
* **Flask**
* **HTML/CSS**
* **Machine Learning Classification Algorithms**

## How It Works

1. Transaction data is collected or generated.
2. Data is analyzed and preprocessed.
3. A machine learning model is trained to classify transactions as *legitimate* or *fraudulent*.
4. The trained model and scaler are saved and integrated into a Flask web application.
5. Users input transaction parameters through the dashboard to receive real-time predictions.

## Use Cases

* Fraud detection research and experimentation
* Learning machine learning model deployment
* Demonstrating real-time financial anomaly detection systems

This project demonstrates how machine learning can be integrated with web technologies to build practical fraud detection systems capable of identifying suspicious transactions quickly and efficiently.
