

import os
import json
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment
)
from azure.identity import DefaultAzureCredential
import joblib
import numpy as np

def connect_to_azure_ml():
    """Connect to Azure ML workspace"""
    
    subscription_id = "ee4291c9-5733-4ee6-9b2d-67a09ef45bd3"
    resource_group = "ml-sara-rg"
    workspace_name = "sara-ml-workspace"
    
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        print(f"Connected to Azure ML workspace: {workspace_name}")
        return ml_client
        
    except Exception as e:
        print(f"Error connecting to Azure ML: {e}")
        return None

def load_feature_names():
    """Load the feature names used during training"""
    try:
        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        return feature_names
    except:
        print("Warning: feature_names.txt not found. Using default features.")
        return None

def preprocess_wine_features(wine_data):
    """Apply the same preprocessing as during training"""
    
    # Create a copy to avoid modifying original
    df = wine_data.copy()
    
    # 1. Alcohol content categories
    df['alcohol_category'] = pd.cut(df['alcohol'], 
                                   bins=[0, 10, 12, 15, 20], 
                                   labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # 2. Acidity ratio
    df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 0.001)
    
    # 3. Sulfur dioxide ratio
    df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 0.001)
    
    # 4. Sugar alcohol interaction
    df['sugar_alcohol_interaction'] = df['residual sugar'] * df['alcohol']
    
    # 5. pH categories
    df['ph_category'] = pd.cut(df['pH'], 
                              bins=[0, 3.0, 3.3, 3.6, 5.0], 
                              labels=['Very_Acidic', 'Acidic', 'Moderate', 'Basic'])
    
    # Add wine_type (assume red wine by default for API)
    if 'wine_type' not in df.columns:
        df['wine_type'] = 'red'
    
    # One-hot encode categorical features
    categorical_features = ['wine_type', 'alcohol_category', 'ph_category']
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    
    return df_encoded

def create_scoring_script():
    """Create scoring script for model deployment"""
    
    scoring_script = """
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
"""
    
    with open("score.py", "w") as f:
        f.write(scoring_script)
    
    print("Fixed scoring script created: score.py")

def test_endpoint_locally():
    """Test the model locally with correct preprocessing"""
    
    print("\nTesting model locally with correct preprocessing...")
    
    try:
        # Load the trained model
        model = joblib.load('best_model.pkl')
        
        # Load feature names
        feature_names = load_feature_names()
        if not feature_names:
            print("Could not load feature names")
            return False
        
        # Create test data (basic wine features)
        test_wine_data = pd.DataFrame([{
            'fixed acidity': 7.4,
            'volatile acidity': 0.7,
            'citric acid': 0.0,
            'residual sugar': 1.9,
            'chlorides': 0.076,
            'free sulfur dioxide': 11.0,
            'total sulfur dioxide': 34.0,
            'density': 0.9978,
            'pH': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4
        }])
        
        # Apply same preprocessing as training
        processed_data = preprocess_wine_features(test_wine_data)
        
        # Add missing features with zeros
        for feature in feature_names:
            if feature not in processed_data.columns:
                processed_data[feature] = 0
        
        # Select features in correct order
        X_test = processed_data[feature_names]
        
        # Make prediction
        prediction = model.predict(X_test)[0]
        probability = model.predict_proba(X_test)[0]
        
        print(f"Test wine prediction: {'Good Wine' if prediction == 1 else 'Regular Wine'}")
        print(f"Confidence: {max(probability):.4f}")
        print(f"Probabilities: Regular={probability[0]:.4f}, Good={probability[1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Local test failed: {e}")
        return False

def create_test_request():
    """Create sample test request for endpoint"""
    
    # Sample wine data for testing (only basic features)
    test_data = {
        "data": [
            {
                "fixed acidity": 7.4,
                "volatile acidity": 0.7,
                "citric acid": 0.0,
                "residual sugar": 1.9,
                "chlorides": 0.076,
                "free sulfur dioxide": 11.0,
                "total sulfur dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            },
            {
                "fixed acidity": 8.1,
                "volatile acidity": 0.28,
                "citric acid": 0.4,
                "residual sugar": 6.9,
                "chlorides": 0.05,
                "free sulfur dioxide": 30.0,
                "total sulfur dioxide": 97.0,
                "density": 0.9951,
                "pH": 3.26,
                "sulphates": 0.44,
                "alcohol": 10.1
            }
        ]
    }
    
    # Save test request
    with open("test_request.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print("Test request created: test_request.json")
    return test_data

def main():
    """Main function to deploy wine quality model"""
    
    print("Wine Quality Model Deployment (Fixed Version)")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ['best_model.pkl', 'feature_names.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing files: {missing_files}")
        print("Run model training first to generate these files.")
        return
    
    # Test model locally first
    if not test_endpoint_locally():
        print("Local test failed. Check your model and preprocessing.")
        return
    
    # Create deployment files
    create_scoring_script()
    create_test_request()
    
    print(f"\n" + "=" * 60)
    print(f"DEPLOYMENT FILES CREATED")
    print(f"=" * 60)
    print(f"Local model test: PASSED")
    print(f"Scoring script: score.py")
    print(f"Test data: test_request.json")
    print(f"Model: Random Forest with {len(load_feature_names())} features")
    
    print(f"\nFiles Created:")
    print(f"   - score.py (fixed scoring script)")
    print(f"   - test_request.json (test data)")
    

    
    print(f"\nModel expects these basic features:")
    basic_features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    for feature in basic_features:
        print(f"   - {feature}")

if __name__ == "__main__":
    main()