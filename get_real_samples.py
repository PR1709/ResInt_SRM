import pandas as pd
import requests
import io
import pickle
import numpy as np

# Load the actual dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
response = requests.get(url)
data = pd.read_csv(io.StringIO(response.text))

print("Dataset shape:", data.shape)
print("Status distribution:")
print(data['status'].value_counts())

# Load the trained model to get feature names
with open('parkinsons_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']

# Get actual samples from the dataset
healthy_samples = data[data['status'] == 0]
parkinsons_samples = data[data['status'] == 1]

print("\nHealthy samples available:", len(healthy_samples))
print("Parkinson's samples available:", len(parkinsons_samples))

# Get a representative healthy sample
healthy_sample = healthy_samples.iloc[0]
parkinsons_sample = parkinsons_samples.iloc[0]

print("\nActual Healthy Sample (status=0):")
healthy_dict = {}
for feature in features:
    value = healthy_sample[feature]
    healthy_dict[feature] = value
    print(f"'{feature}': {value},")

print("\nActual Parkinson's Sample (status=1):")
parkinsons_dict = {}
for feature in features:
    value = parkinsons_sample[feature]
    parkinsons_dict[feature] = value
    print(f"'{feature}': {value},")

# Test these samples with the model
def test_sample(sample_data, sample_name):
    feature_array = []
    for feature in features:
        feature_array.append(sample_data[feature])
    
    feature_array = np.array(feature_array).reshape(1, -1)
    scaled_features = scaler.transform(feature_array)
    
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    
    print(f"\n{sample_name} Sample Test:")
    print(f"Prediction: {prediction} ({'Healthy' if prediction == 0 else 'Parkinsons'})")
    print(f"Healthy probability: {probabilities[0]:.4f} ({probabilities[0]*100:.2f}%)")
    print(f"Parkinsons probability: {probabilities[1]:.4f} ({probabilities[1]*100:.2f}%)")

test_sample(healthy_dict, "Real Healthy")
test_sample(parkinsons_dict, "Real Parkinsons")
