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
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dataset_info = {}
        
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
        
        # Store dataset information
        self.dataset_info = {
            'total_samples': len(data),
            'healthy_samples': len(data[data['status'] == 0]),
            'parkinsons_samples': len(data[data['status'] == 1]),
            'features': len(self.feature_names)
        }
        
        return X_scaled, y
    
    def get_available_models(self):
        return {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
    
    def train_selected_models(self, X, y, selected_models):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        available_models = self.get_available_models()
        results = {}
        
        for model_name in selected_models:
            if model_name in available_models:
                model = available_models[model_name]
                
                # Train the model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Get probabilities if available
                try:
                    y_proba_train = model.predict_proba(self.X_train)[:, 1]
                    y_proba_test = model.predict_proba(self.X_test)[:, 1]
                except:
                    y_proba_train = y_pred_train
                    y_proba_test = y_pred_test
                
                # Calculate metrics
                results[model_name] = {
                    'model': model,
                    'train_accuracy': accuracy_score(self.y_train, y_pred_train),
                    'test_accuracy': accuracy_score(self.y_test, y_pred_test),
                    'train_precision': precision_score(self.y_train, y_pred_train),
                    'test_precision': precision_score(self.y_test, y_pred_test),
                    'train_recall': recall_score(self.y_train, y_pred_train),
                    'test_recall': recall_score(self.y_test, y_pred_test),
                    'train_f1': f1_score(self.y_train, y_pred_train),
                    'test_f1': f1_score(self.y_test, y_pred_test),
                    'train_auc': roc_auc_score(self.y_train, y_proba_train),
                    'test_auc': roc_auc_score(self.y_test, y_proba_test)
                }
                
                # Store the trained model
                self.trained_models[model_name] = model
        
        return results
    
    def save_models(self, filename='parkinsons_models.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'models': self.trained_models, 
                'scaler': self.scaler, 
                'features': self.feature_names,
                'dataset_info': self.dataset_info
            }, f)
    
    def load_models(self, filename='parkinsons_models.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.trained_models = data.get('models', {})
                self.scaler = data['scaler']
                self.feature_names = data['features']
                self.dataset_info = data.get('dataset_info', {})
            return True
        return False
    
    def predict_with_model(self, features, model_name):
        if model_name not in self.trained_models:
            return None, None
        
        model = self.trained_models[model_name]
        features_scaled = self.scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        try:
            probability = model.predict_proba(features_scaled)[0]
        except:
            probability = [1-prediction, prediction]
            
        return prediction, probability

detector = ParkinsonsDetector()

# Initialize with dataset
print("Loading dataset...")
data = detector.load_data()
X, y = detector.preprocess_data(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def get_available_models():
    models = list(detector.get_available_models().keys())
    return jsonify({'models': models})

@app.route('/train', methods=['POST'])
def train_models():
    try:
        selected_models = request.json.get('models', [])
        if not selected_models:
            return jsonify({'error': 'No models selected'}), 400
        
        results = detector.train_selected_models(X, y, selected_models)
        
        # Format results for frontend
        formatted_results = {}
        for model_name, metrics in results.items():
            formatted_results[model_name] = {
                'train_accuracy': round(metrics['train_accuracy'], 4),
                'test_accuracy': round(metrics['test_accuracy'], 4),
                'train_precision': round(metrics['train_precision'], 4),
                'test_precision': round(metrics['test_precision'], 4),
                'train_recall': round(metrics['train_recall'], 4),
                'test_recall': round(metrics['test_recall'], 4),
                'train_f1': round(metrics['train_f1'], 4),
                'test_f1': round(metrics['test_f1'], 4),
                'train_auc': round(metrics['train_auc'], 4),
                'test_auc': round(metrics['test_auc'], 4)
            }
        
        detector.save_models()
        
        return jsonify({
            'results': formatted_results,
            'dataset_info': detector.dataset_info,
            'training_samples': len(detector.y_train),
            'testing_samples': len(detector.y_test)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_name = data.get('model', 'Random Forest')
        features = []
        
        for feature in detector.feature_names:
            value = float(data.get(feature, 0))
            features.append(value)
        
        prediction, probability = detector.predict_with_model(features, model_name)
        
        if prediction is None:
            return jsonify({'error': f'Model {model_name} not trained'}), 400
        
        return jsonify({
            'model': model_name,
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