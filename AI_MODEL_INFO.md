# AI/ML Model Information & Dark Mode Implementation

## 🤖 Machine Learning Model Details

### Algorithm Used: K-Nearest Neighbors (KNN)

The Parkinson's Disease Detection System uses a **K-Nearest Neighbors (KNN)** classifier as its final model after comprehensive testing and hyperparameter optimization.

### Model Selection Process

1. **Initial Training Phase**: 6 different algorithms were trained and evaluated:
   - **Logistic Regression**
   - **Decision Tree Classifier**
   - **Naive Bayes (Gaussian)**
   - **K-Nearest Neighbors (KNN)**
   - **Support Vector Machine (SVM)**
   - **Random Forest Classifier**

2. **Model Evaluation Metrics**:
   - Accuracy Score
   - Precision Score
   - Recall Score
   - F1 Score
   - ROC AUC Score

3. **Best Model Selection**: The model with the highest accuracy was initially selected

4. **Hyperparameter Tuning**: KNN was further optimized using GridSearchCV with 5-fold cross-validation

### Final KNN Model Parameters

- **n_neighbors**: 11 (number of nearest neighbors to consider)
- **weights**: 'distance' (closer neighbors have more influence)
- **metric**: 'manhattan' (distance calculation method)
- **algorithm**: 'auto' (automatically chooses the best algorithm)

### Dataset Information

- **Source**: UCI Machine Learning Repository - Parkinson's Disease Dataset
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
- **Total Samples**: 195 voice recordings
- **Healthy Samples**: 48 (status=0)
- **Parkinson's Samples**: 147 (status=1)
- **Features**: 22 voice biomarkers

### Voice Features (Biomarkers)

The model analyzes 22 different voice characteristics:

1. **Fundamental Frequency Measures**:
   - MDVP:Fo(Hz) - Average vocal fundamental frequency
   - MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
   - MDVP:Flo(Hz) - Minimum vocal fundamental frequency

2. **Jitter Measures** (frequency variation):
   - MDVP:Jitter(%) - Percentage jitter
   - MDVP:Jitter(Abs) - Absolute jitter
   - MDVP:RAP - Relative average perturbation
   - MDVP:PPQ - Five-point period perturbation quotient
   - Jitter:DDP - Average absolute difference of differences

3. **Shimmer Measures** (amplitude variation):
   - MDVP:Shimmer - Shimmer percentage
   - MDVP:Shimmer(dB) - Shimmer in decibels
   - Shimmer:APQ3 - Three-point amplitude perturbation quotient
   - Shimmer:APQ5 - Five-point amplitude perturbation quotient
   - MDVP:APQ - Amplitude perturbation quotient
   - Shimmer:DDA - Average absolute difference of differences

4. **Noise Measures**:
   - NHR - Noise-to-harmonics ratio
   - HNR - Harmonics-to-noise ratio

5. **Nonlinear Measures**:
   - RPDE - Recurrence period density entropy
   - DFA - Detrended fluctuation analysis
   - spread1 - Fundamental frequency variation
   - spread2 - Fundamental frequency variation
   - D2 - Correlation dimension
   - PPE - Pitch period entropy

### Data Preprocessing

- **Feature Scaling**: StandardScaler normalization applied to all features
- **Train-Test Split**: 80% training, 20% testing (random_state=42)
- **Cross-Validation**: 5-fold CV used for hyperparameter tuning

### Model Performance

The KNN model was selected as the best performer after hyperparameter tuning, achieving optimal accuracy through grid search optimization.

## 🌙 Dark Mode Implementation

### Features

1. **Toggle Button**: Moon/Sun icon in the top-right corner
2. **Persistent Theme**: User preference saved in localStorage
3. **Smooth Transitions**: 0.3s ease transitions for all elements
4. **Comprehensive Coverage**: All UI elements support both themes

### Technical Implementation

#### CSS Variables System
```css
:root {
    --bg-gradient-light: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --bg-gradient-dark: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    --container-bg-light: rgba(255, 255, 255, 0.95);
    --container-bg-dark: rgba(30, 30, 50, 0.95);
    /* ... more variables */
}
```

#### JavaScript Theme Management
- `toggleTheme()`: Switches between light/dark modes
- `loadTheme()`: Loads saved preference on page load
- `localStorage`: Persists user theme choice

#### Dark Mode Color Scheme
- **Background**: Dark blue gradient (#1a1a2e to #16213e)
- **Containers**: Semi-transparent dark blue (#2a2a3e)
- **Text**: Light colors (#e0e0e0, #b0b0b0)
- **Inputs**: Dark backgrounds with light borders
- **Accents**: Brighter blue tones (#8a9cff)

### Accessibility Features

- High contrast ratios in both themes
- Smooth transitions prevent jarring changes
- Clear visual indicators for interactive elements
- Consistent color usage across all components

## 🚀 Usage Instructions

1. **Theme Toggle**: Click the moon/sun button in the top-right corner
2. **Sample Data**: Use "Load Healthy Sample" or "Load Parkinson's Sample" buttons
3. **File Upload**: Click "Load from File" to upload CSV, JSON, or TXT files
4. **Analysis**: Click "Analyze Voice Features" to get predictions
5. **Results**: View probability percentages and confidence levels

## 📊 Model Information Display

The interface now includes a dedicated "AI Model Information" section showing:
- Algorithm type and parameters
- Feature count and dataset information
- Training methodology
- Preprocessing details

This provides transparency about the AI system's capabilities and limitations.
