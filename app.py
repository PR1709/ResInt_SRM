import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import os
from flask import Flask, render_template, request, jsonify
import requests
import io

app = Flask(__name__)

class ParkinsonsDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []
        
    def load_data(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        response = requests.get(url)
        data = pd.read_csv(io.StringIO(response.text))
        return data
    
    def preprocess_data(self, data):
        data = data.drop('name', axis=1)
        X = data.drop('status', axis=1)
        y = data['status']
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def train_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba)
            }
        
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.models = results
        
        return results, X_test, y_test
    
    def hyperparameter_tuning(self, X, y):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)
        
        self.best_model = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
    
    def save_model(self, filename='parkinsons_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({'model': self.best_model, 'scaler': self.scaler, 'features': self.feature_names}, f)
    
    def load_model(self, filename='parkinsons_model.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.best_model = data['model']
                self.scaler = data['scaler']
                self.feature_names = data['features']
            return True
        return False
    
    def predict(self, features):
        if self.best_model is None:
            return None
        features_scaled = self.scaler.transform([features])
        prediction = self.best_model.predict(features_scaled)[0]
        probability = self.best_model.predict_proba(features_scaled)[0] if hasattr(self.best_model, 'predict_proba') else [1-prediction, prediction]
        return prediction, probability

detector = ParkinsonsDetector()

if not detector.load_model():
    print("Training new model...")
    data = detector.load_data()
    X, y = detector.preprocess_data(data)
    results, X_test, y_test = detector.train_models(X, y)
    best_params, best_score = detector.hyperparameter_tuning(X, y)
    detector.save_model()
    print(f"Model trained and saved. Best accuracy: {best_score:.4f}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        for feature in detector.feature_names:
            value = float(request.json.get(feature, 0))
            features.append(value)
        
        prediction, probability = detector.predict(features)
        
        return jsonify({
            'prediction': int(prediction),
            'probability': {
                'healthy': float(probability[0]),
                'parkinsons': float(probability[1])
            },
            'confidence': float(max(probability))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/features')
def get_features():
    return jsonify({'features': detector.feature_names})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read file content
        content = file.read().decode('utf-8')
        data = {}

        if file.filename.endswith('.json'):
            import json
            data = json.loads(content)
        elif file.filename.endswith('.csv'):
            lines = content.strip().split('\n')
            if len(lines) >= 2:
                headers = [h.strip().replace('"', '') for h in lines[0].split(',')]
                values = [v.strip().replace('"', '') for v in lines[1].split(',')]
                for i, header in enumerate(headers):
                    if i < len(values):
                        try:
                            data[header] = float(values[i])
                        except ValueError:
                            data[header] = values[i]

        # Filter only valid features
        valid_data = {}
        for feature in detector.feature_names:
            if feature in data:
                valid_data[feature] = data[feature]

        return jsonify({
            'data': valid_data,
            'loaded_features': len(valid_data),
            'total_features': len(detector.feature_names)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)