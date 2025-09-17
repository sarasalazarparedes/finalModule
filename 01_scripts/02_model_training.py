
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')

def setup_mlflow():
    """Configure MLflow for experiment tracking"""
    
    # Set tracking URI (local for now)
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set or create experiment
    experiment_name = "wine-quality-sara-clases"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name}")
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    return experiment_name

def load_processed_data():
    """Load the preprocessed training and testing data"""
    
    try:
        X_train = pd.read_csv('X_train.csv')
        X_test = pd.read_csv('X_test.csv')
        y_train = pd.read_csv('y_train.csv').iloc[:, 0]  # Get first column as Series
        y_test = pd.read_csv('y_test.csv').iloc[:, 0]
        
        print(f"Data loaded successfully:")
        print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"  Class distribution (train): {y_train.value_counts().values}")
        print(f"  Class distribution (test): {y_test.value_counts().values}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure you've run 01_dataset_preparation.py first!")
        return None, None, None, None

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and return metrics"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
    }
    
    # Add AUC if probabilities available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics, y_pred, y_pred_proba

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with MLflow tracking"""
    
    with mlflow.start_run(run_name="RandomForest_WineQuality"):
        print(f"\nTraining Random Forest Classifier...")
        
        # Hyperparameters
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("dataset_size", len(X_train))
        
        # Train model
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Random Forest - Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('rf_feature_importance.png')
        plt.close()
        
        # Log model
        mlflow.sklearn.log_model(rf_model, "model")
        
        # Save model locally
        joblib.dump(rf_model, 'random_forest_model.pkl')
        
        return rf_model, metrics

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with MLflow tracking"""
    
    with mlflow.start_run(run_name="LogisticRegression_WineQuality"):
        print(f"\nTraining Logistic Regression...")
        
        # Hyperparameters
        params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear',
            'max_iter': 1000,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("dataset_size", len(X_train))
        
        # Train model
        lr_model = LogisticRegression(**params)
        lr_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(lr_model, "model")
        
        # Save model locally
        joblib.dump(lr_model, 'logistic_regression_model.pkl')
        
        return lr_model, metrics

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting with MLflow tracking"""
    
    with mlflow.start_run(run_name="GradientBoosting_WineQuality"):
        print(f"\nTraining Gradient Boosting Classifier...")
        
        # Hyperparameters
        params = {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.log_param("dataset_size", len(X_train))
        
        # Train model
        gb_model = GradientBoostingClassifier(**params)
        gb_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(gb_model, "model")
        
        # Save model locally
        joblib.dump(gb_model, 'gradient_boosting_model.pkl')
        
        return gb_model, metrics

def compare_models(model_results):
    """Compare all trained models and select the best one"""
    
    print(f"\n" + "=" * 50)
    print(f"MODEL COMPARISON")
    print(f"=" * 50)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, results in model_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics.get('roc_auc', 'N/A')
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4))
    
    # Save comparison
    comparison_df.to_csv('model_comparison.csv', index=False)
    
    # Identify best model based on F1-Score
    best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
    best_model = model_results[best_model_name]['model']
    best_metrics = model_results[best_model_name]['metrics']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"   F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   ROC-AUC: {best_metrics.get('roc_auc', 'N/A')}")
    
    # Save best model with special name
    joblib.dump(best_model, 'best_model.pkl')
    
    return best_model_name, best_model, best_metrics, comparison_df

def create_evaluation_plots(comparison_df, best_model_name, best_model, X_test, y_test):
    """Create comprehensive evaluation visualizations"""
    
    print(f"\nCreating evaluation visualizations...")
    
    # Predictions for best model
    y_pred_best = best_model.predict(X_test)
    y_pred_proba_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Model Comparison
    comparison_metrics = comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
    comparison_metrics.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. ROC Curve (if available)
    if y_pred_proba_best is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
        roc_auc = roc_auc_score(y_test, y_pred_proba_best)
        axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'{best_model_name} (AUC = {roc_auc:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 4. Prediction Distribution
    if y_pred_proba_best is not None:
        axes[1, 1].hist(y_pred_proba_best[y_test == 0], bins=20, alpha=0.5, label='Regular Wine', color='red')
        axes[1, 1].hist(y_pred_proba_best[y_test == 1], bins=20, alpha=0.5, label='Good Wine', color='green')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model_evaluation_complete.png', dpi=300, bbox_inches='tight')
    print(f"   - model_evaluation_complete.png saved")
    plt.close()

def register_best_model(best_model_name, best_model, best_metrics, X_train):
    """Register best model in MLflow"""
    
    with mlflow.start_run(run_name=f"BEST_MODEL_{best_model_name}"):
        mlflow.log_param("model_type", best_model_name)
        mlflow.log_param("status", "CHAMPION")
        mlflow.log_param("dataset_size", len(X_train))
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_model, "champion_model")
        
        # Log evaluation artifacts
        mlflow.log_artifact('model_evaluation_complete.png')
        mlflow.log_artifact('model_comparison.csv')
        
        print(f"\nBest model registered in MLflow")

def main():
    """Main function to run the complete model training pipeline"""
    
    print("Wine Quality Model Training with MLflow")
    print("=" * 50)
    
    # Setup MLflow
    setup_mlflow()
    
    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()
    if X_train is None:
        return
    
    print(f"\n" + "=" * 50)
    print(f"TRAINING MULTIPLE MODELS")
    print(f"=" * 50)
    
    # Store results
    model_results = {}
    
    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    model_results['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    # Train Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
    model_results['Logistic Regression'] = {'model': lr_model, 'metrics': lr_metrics}
    
    # Train Gradient Boosting
    gb_model, gb_metrics = train_gradient_boosting(X_train, X_test, y_train, y_test)
    model_results['Gradient Boosting'] = {'model': gb_model, 'metrics': gb_metrics}
    
    # Compare models and select best
    best_model_name, best_model, best_metrics, comparison_df = compare_models(model_results)
    
    # Create evaluation plots
    create_evaluation_plots(comparison_df, best_model_name, best_model, X_test, y_test)
    
    # Register best model
    register_best_model(best_model_name, best_model, best_metrics, X_train)
    
    # Print detailed evaluation
    print(f"\n" + "=" * 50)
    print(f"DETAILED EVALUATION - {best_model_name}")
    print(f"=" * 50)
    
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_best))
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"MODEL TRAINING SUMMARY")
    print(f"=" * 60)
    print(f"Problem: Binary Classification (Wine Quality)")
    print(f"Training Data: {len(X_train):,} samples")
    print(f"Models Trained: {len(model_results)}")
    print(f"Best Model: {best_model_name}")
    print(f"Best F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"MLflow Tracking: ./mlruns")
    
    print(f"\nFiles Created:")
    print(f"   - random_forest_model.pkl")
    print(f"   - logistic_regression_model.pkl")
    print(f"   - gradient_boosting_model.pkl")
    print(f"   - best_model.pkl ({best_model_name})")
    print(f"   - model_comparison.csv")
    print(f"   - model_evaluation_complete.png")
    print(f"   - MLflow experiment runs")


if __name__ == "__main__":
    main()