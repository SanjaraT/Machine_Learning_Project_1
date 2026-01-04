📌 Overview

    This project applies multiple machine learning models to classify gamma-ray events and hadronic background events using the MAGIC Gamma Telescope dataset from the UCI Machine Learning Repository.
    The work demonstrates a complete ML pipeline from data preprocessing to model evaluation.

📊 Dataset

    Source: UCI ML Repository
    Samples: 19,020
    Features: 10 numerical attributes
    Classes:
    1 → Gamma
    0 → Hadron

🔍 Preprocessing

    Label encoding
    Stratified train/validation/test split (70/15/15)
    Feature normalization (StandardScaler)
    Training set balancing to address class imbalance

🤖 Models Used

    K-Nearest Neighbors (KNN)
    Gaussian Naive Bayes
    Logistic Regression
    Support Vector Machine (RBF)
    Neural Network (Sequential)

🧠 Neural Network

    Architecture: 64 → 32 → 1
    Activations: ReLU, Sigmoid
    Optimizer: Adam
    Loss: Binary Cross-Entropy
    Early stopping applied

📈 Evaluation

    Models were evaluated using:
    Accuracy
    Precision, Recall, F1-score
    Confusion Matrix
    Classification Report

🛠️ Tools

    Python, NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow/Keras
