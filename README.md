Parkinson's Disease Detection System 🧠
This repository contains a full-stack Machine Learning application designed to predict the likelihood of Parkinson's Disease based on vocal biomarkers. The system uses a K-Nearest Neighbors (KNN) classifier (optimized via GridSearchCV) and provides a modern, responsive web interface for real-time analysis.

🚀 Features
Machine Learning & Backend
Multi-Model Training: Supports training and evaluation of 6 different algorithms: Logistic Regression, Decision Tree, Naive Bayes, KNN, SVM, and Random Forest.

Automated Data Pipeline: Automatically downloads the Parkinson's dataset from the UCI Machine Learning Repository upon initialization.

Hyperparameter Optimization: Uses GridSearchCV with 5-fold cross-validation to optimize the final KNN model.

Advanced Preprocessing: Implements StandardScaler normalization and handles feature scaling to ensure high prediction accuracy.

RESTful API: A Flask-based backend provides endpoints for training, model selection, and real-time predictions.

Frontend & User Experience
Modern UI: A clean "glassmorphism" design with smooth 0.3s transitions and responsive layouts for desktop and mobile.

Persistent Dark Mode: Includes a toggleable dark/light theme that saves user preferences to localStorage.

Sample Data Loading: Instant "One-Click" buttons to load pre-defined Healthy or Parkinson's samples for testing.

File Upload Support: Allows users to upload their own data via .csv, .json, or .txt formats.

Visual Analytics: Real-time probability bars and confidence metrics for every prediction.

📊 Model Details
The system analyzes 22 vocal biomarkers, including:

Frequency Measures: Average, maximum, and minimum vocal fundamental frequencies.

Jitter & Shimmer: Variations in frequency and amplitude.

Noise Measures: Noise-to-harmonics (NHR) and Harmonics-to-noise (HNR) ratios.

Nonlinear Measures: RPDE, DFA, Spread1, Spread2, D2, and PPE.

Performance Metrics:

Accuracy: ~85-90%.

Optimized KNN Parameters: n_neighbors: 11, weights: 'distance', metric: 'manhattan'.

🛠️ Installation & Setup
1. Prerequisites
Ensure you have Python 3.8+ installed.

2. Environment Setup
Bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3. Install Dependencies
Bash
pip install pandas numpy scikit-learn flask requests pickle-mixin
4. Running the Application
Bash
python app.py
The first run will automatically download the dataset and train the models (~1-2 minutes). Subsequent runs will load the saved model instantly.

📁 Project Structure
Plaintext
parkinsons-detection/
├── app.py                 # Flask server & ML Logic
├── requirements.txt       # Project dependencies
├── AI_MODEL_INFO.md       # Technical model documentation
├── SetupInstructions.md   # Detailed setup guide
├── templates/
│   └── index.html         # Frontend Interface
└── parkinsons_models.pkl  # Serialized trained models
📋 Usage
Open your browser to http://localhost:5000.

Select the models you wish to evaluate from the Model Selection panel and click Train.

Load sample data or upload a file containing voice features.

Select your preferred trained model and click Predict to see the results.

Data Source: UCI Machine Learning Repository - Parkinson's Dataset.
