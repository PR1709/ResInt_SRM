import pickle
import numpy as np

# Load the trained model
with open('parkinsons_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']

print("Feature names from model:", features)
print("Number of features:", len(features))

# Sample data from the HTML file
healthy_sample = {
    'MDVP:Fo(Hz)': 119.99200, 'MDVP:Fhi(Hz)': 157.30200, 'MDVP:Flo(Hz)': 74.99700,
    'MDVP:Jitter(%)': 0.00784, 'MDVP:Jitter(Abs)': 0.00007, 'MDVP:RAP': 0.00370,
    'MDVP:PPQ': 0.00554, 'Jitter:DDP': 0.01109, 'MDVP:Shimmer': 0.04374,
    'MDVP:Shimmer(dB)': 0.42600, 'Shimmer:APQ3': 0.02182, 'Shimmer:APQ5': 0.03130,
    'MDVP:APQ': 0.02971, 'Shimmer:DDA': 0.06545, 'NHR': 0.02211, 'HNR': 21.03300,
    'RPDE': 0.414783, 'DFA': 0.815285, 'spread1': -4.813031, 'spread2': 0.266482,
    'D2': 2.301442, 'PPE': 0.284654
}

parkinsons_sample = {
    'MDVP:Fo(Hz)': 197.07600, 'MDVP:Fhi(Hz)': 206.89600, 'MDVP:Flo(Hz)': 192.05500,
    'MDVP:Jitter(%)': 0.00289, 'MDVP:Jitter(Abs)': 0.00001, 'MDVP:RAP': 0.00166,
    'MDVP:PPQ': 0.00168, 'Jitter:DDP': 0.00498, 'MDVP:Shimmer': 0.01098,
    'MDVP:Shimmer(dB)': 0.09700, 'Shimmer:APQ3': 0.00563, 'Shimmer:APQ5': 0.00680,
    'MDVP:APQ': 0.00802, 'Shimmer:DDA': 0.01689, 'NHR': 0.00339, 'HNR': 26.77500,
    'RPDE': 0.422229, 'DFA': 0.741367, 'spread1': -7.348300, 'spread2': 0.177551,
    'D2': 1.743867, 'PPE': 0.085569
}

def test_sample(sample_data, sample_name):
    # Convert to feature array in the correct order
    feature_array = []
    for feature in features:
        if feature in sample_data:
            feature_array.append(sample_data[feature])
        else:
            print(f"Missing feature: {feature}")
            return
    
    # Scale the features
    feature_array = np.array(feature_array).reshape(1, -1)
    scaled_features = scaler.transform(feature_array)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    
    print(f"\n{sample_name} Sample Results:")
    print(f"Prediction: {prediction} ({'Healthy' if prediction == 0 else 'Parkinsons'})")
    print(f"Healthy probability: {probabilities[0]:.4f} ({probabilities[0]*100:.2f}%)")
    print(f"Parkinsons probability: {probabilities[1]:.4f} ({probabilities[1]*100:.2f}%)")
    print(f"Confidence: {max(probabilities):.4f} ({max(probabilities)*100:.2f}%)")

# Test both samples
test_sample(healthy_sample, "Healthy")
test_sample(parkinsons_sample, "Parkinsons")
