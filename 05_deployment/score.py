
import json
import joblib
import numpy as np
import pandas as pd
from io import StringIO

def init():
    global model, feature_names
    try:
        # Load model
        model = joblib.load('best_model.pkl')
        
        # Load feature names
        try:
            with open('feature_names.txt', 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
        except:
            feature_names = None
        
        print("Model and features loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        feature_names = None

def preprocess_wine_features(wine_data):
    '''Apply the same preprocessing as during training'''
    
    df = wine_data.copy()
    
    # Feature engineering
    df['alcohol_category'] = pd.cut(df['alcohol'], 
                                   bins=[0, 10, 12, 15, 20], 
                                   labels=['Low', 'Medium', 'High', 'Very_High'])
    
    df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 0.001)
    df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 0.001)
    df['sugar_alcohol_interaction'] = df['residual sugar'] * df['alcohol']
    
    df['ph_category'] = pd.cut(df['pH'], 
                              bins=[0, 3.0, 3.3, 3.6, 5.0], 
                              labels=['Very_Acidic', 'Acidic', 'Moderate', 'Basic'])
    
    # Add wine_type if not present
    if 'wine_type' not in df.columns:
        df['wine_type'] = 'red'
    
    # One-hot encode
    categorical_features = ['wine_type', 'alcohol_category', 'ph_category']
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    
    return df_encoded

def run(raw_data):
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Convert to DataFrame
        if isinstance(data, dict) and 'data' in data:
            input_data = pd.DataFrame(data['data'])
        elif isinstance(data, list):
            input_data = pd.DataFrame(data)
        else:
            input_data = pd.DataFrame([data])
        
        # Basic features expected
        basic_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        
        # Check basic features
        missing_basic = [f for f in basic_features if f not in input_data.columns]
        if missing_basic:
            return json.dumps({
                "error": f"Missing basic features: {missing_basic}",
                "required_features": basic_features
            })
        
        # Apply preprocessing
        processed_data = preprocess_wine_features(input_data)
        
        # Align with training features
        if feature_names:
            # Add missing columns with zeros
            for feature in feature_names:
                if feature not in processed_data.columns:
                    processed_data[feature] = 0
            
            # Select only training features in correct order
            X = processed_data[feature_names]
        else:
            X = processed_data
        
        # Make predictions
        if model is not None:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "wine_sample": i + 1,
                    "prediction": int(pred),
                    "prediction_label": "Good Wine" if pred == 1 else "Regular Wine",
                    "confidence": float(max(prob)),
                    "probability_regular": float(prob[0]),
                    "probability_good": float(prob[1])
                })
            
            return json.dumps({
                "predictions": results,
                "model": "Random Forest Wine Quality Classifier",
                "version": "1.0"
            })
        else:
            return json.dumps({"error": "Model not loaded properly"})
            
    except Exception as e:
        return json.dumps({"error": f"Prediction error: {str(e)}"})
