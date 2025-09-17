#!/usr/bin/env python3
"""
Wine Quality Dataset Preparation for MLOps Project
Dataset: Wine Quality (Red + White wines) - 6,497 total records
Problem: Binary classification (Good wine vs Regular wine)
Source: UCI Machine Learning Repository (FREE)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_wine_dataset():
    """Load and combine red and white wine datasets"""
    
    # URLs for free UCI datasets
    red_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    white_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    
    try:
        print("Loading red wine dataset...")
        red_wine = pd.read_csv(red_wine_url, sep=';')
        
        print("Loading white wine dataset...")
        white_wine = pd.read_csv(white_wine_url, sep=';')
        
        # Add wine type feature
        red_wine['wine_type'] = 'red'
        white_wine['wine_type'] = 'white'
        
        # Combine datasets
        combined_wine = pd.concat([red_wine, white_wine], ignore_index=True)
        
        print(f"Dataset loaded successfully!")
        print(f"   Red wine samples: {len(red_wine)}")
        print(f"   White wine samples: {len(white_wine)}")
        print(f"   Total samples: {len(combined_wine)}")
        
        return combined_wine
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset(df):
    """Explore and analyze the dataset"""
    
    print("\n" + "=" * 50)
    print("DATASET EXPLORATION")
    print("=" * 50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    print(f"\nQuality distribution:")
    quality_counts = df['quality'].value_counts().sort_index()
    print(quality_counts)
    
    print(f"\nWine type distribution:")
    print(df['wine_type'].value_counts())
    
    return df

def create_binary_target(df):
    """Convert quality scores to binary classification problem"""
    
    # Strategy: Quality >= 6 = Good wine (1), Quality < 6 = Regular wine (0)
    df['quality_binary'] = (df['quality'] >= 6).astype(int)
    
    print(f"\nBinary classification target created:")
    print(f"Good wine (quality >= 6): {sum(df['quality_binary'] == 1)} samples")
    print(f"Regular wine (quality < 6): {sum(df['quality_binary'] == 0)} samples")
    
    # Check class balance
    class_balance = df['quality_binary'].value_counts(normalize=True)
    print(f"\nClass balance:")
    print(f"Regular wine (0): {class_balance[0]:.2%}")
    print(f"Good wine (1): {class_balance[1]:.2%}")
    
    return df

def feature_engineering(df):
    """Create additional features for better model performance"""
    
    print(f"\nFeature Engineering:")
    
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
    
    new_features = ['alcohol_category', 'acidity_ratio', 'sulfur_ratio', 
                   'sugar_alcohol_interaction', 'ph_category']
    
    print(f"Created {len(new_features)} new features:")
    for feature in new_features:
        print(f"   - {feature}")
    
    return df

def preprocess_data(df):
    """Prepare data for machine learning"""
    
    print(f"\nData Preprocessing:")
    
    # Select features for modeling
    numeric_features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'acidity_ratio', 'sulfur_ratio', 
        'sugar_alcohol_interaction'
    ]
    
    categorical_features = ['wine_type', 'alcohol_category', 'ph_category']
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    
    # Get all feature columns (excluding target variables)
    feature_columns = numeric_features + [col for col in df_encoded.columns 
                                        if any(cat in col for cat in categorical_features)]
    
    # Features and target
    X = df_encoded[feature_columns]
    y = df_encoded['quality_binary']
    
    print(f"Features prepared: {len(feature_columns)} total features")
    print(f"   Numeric features: {len(numeric_features)}")
    print(f"   Categorical features (one-hot encoded): {len(feature_columns) - len(numeric_features)}")
    
    return X, y, feature_columns

def split_and_scale_data(X, y):
    """Split data and apply scaling"""
    
    print(f"\nTrain-Test Split:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Data split completed:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    print(f"\nFeature Scaling:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, feature_names, original_df):
    """Save all processed data to files"""
    
    print(f"\nSaving processed data:")
    
    # Create DataFrames for easy saving
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    
    # Save to CSV files
    X_train_df.to_csv('X_train.csv', index=False)
    X_test_df.to_csv('X_test.csv', index=False)
    y_train_df.to_csv('y_train.csv', index=False)
    y_test_df.to_csv('y_test.csv', index=False)
    
    # Save original dataset
    original_df.to_csv('wine_quality_complete.csv', index=False)
    
    # Save feature names
    with open('feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"Data saved successfully:")
    print(f"   - wine_quality_complete.csv (full dataset)")
    print(f"   - X_train.csv, X_test.csv (features)")
    print(f"   - y_train.csv, y_test.csv (targets)")
    print(f"   - feature_names.txt (feature list)")

def create_visualizations(df):
    """Create and save dataset visualizations"""
    
    print(f"\nCreating visualizations:")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Quality distribution
    axes[0, 0].hist(df['quality'], bins=range(3, 11), alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Wine Quality Distribution')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Binary target distribution
    df['quality_binary'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Binary Classification Target')
    axes[0, 1].set_xlabel('Wine Quality (0=Regular, 1=Good)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 3. Wine type vs Quality
    wine_quality_by_type = df.groupby(['wine_type', 'quality_binary']).size().unstack()
    wine_quality_by_type.plot(kind='bar', ax=axes[1, 0], color=['lightcoral', 'lightgreen'])
    axes[1, 0].set_title('Wine Quality by Type')
    axes[1, 0].set_xlabel('Wine Type')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend(['Regular', 'Good'])
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # 4. Alcohol vs Quality
    df.boxplot(column='alcohol', by='quality_binary', ax=axes[1, 1])
    axes[1, 1].set_title('Alcohol Content by Wine Quality')
    axes[1, 1].set_xlabel('Wine Quality (0=Regular, 1=Good)')
    axes[1, 1].set_ylabel('Alcohol Content (%)')
    
    plt.tight_layout()
    plt.savefig('wine_dataset_exploration.png', dpi=300, bbox_inches='tight')
    print(f"   - wine_dataset_exploration.png saved")
    plt.close()

def main():
    """Main function to run the complete dataset preparation"""
    
    print("Wine Quality Dataset - MLOps Project")
    print("=" * 50)
    
    # Load dataset
    wine_data = load_wine_dataset()
    if wine_data is None:
        return
    
    # Explore dataset
    wine_data = explore_dataset(wine_data)
    
    # Create binary target
    wine_data = create_binary_target(wine_data)
    
    # Feature engineering
    wine_data = feature_engineering(wine_data)
    
    # Preprocess data
    X, y, feature_names = preprocess_data(wine_data)
    
    # Split and scale
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Save processed data
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, feature_names, wine_data)
    
    # Create visualizations
    create_visualizations(wine_data)
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"WINE QUALITY DATASET - SUMMARY REPORT")
    print(f"=" * 60)
    print(f"Dataset Size: {len(wine_data):,} samples")
    print(f"Problem Type: Binary Classification")
    print(f"Features: {len(feature_names)} total features")
    print(f"Wine Types: Red + White")
    print(f"Class Balance: {y.value_counts()[0]} Regular, {y.value_counts()[1]} Good")
    print(f"Train/Test Split: {len(X_train_scaled)}/{len(X_test_scaled)} samples")
    print(f"Ready for Machine Learning!")
    
    print(f"\nFiles created:")
    print(f"   - wine_quality_complete.csv")
    print(f"   - X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print(f"   - feature_names.txt")
    print(f"   - wine_dataset_exploration.png")
    
    print(f"\nNext steps:")
    print(f"   1. Run: python 02_model_training.py")
    print(f"   2. Create Azure ML pipeline")
    print(f"   3. Deploy best model")

if __name__ == "__main__":
    main()