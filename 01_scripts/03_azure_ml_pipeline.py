

import os
import pandas as pd
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Data, Model, Environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import dsl
from azure.identity import DefaultAzureCredential
import joblib

def connect_to_azure_ml():
    """Connect to Azure ML workspace"""
    
    # Azure configuration
    subscription_id = "ee4291c9-5733-4ee6-9b2d-67a09ef45bd3"
    resource_group = "ml-sara-rg"
    workspace_name = "sara-ml-workspace"
    
    try:
        # Create MLClient
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

def create_data_asset(ml_client):
    """Create and register data asset in Azure ML"""
    
    try:
        # Check if wine dataset already exists
        try:
            data_asset = ml_client.data.get(name="wine-quality-dataset", version="1")
            print(f"Data asset already exists: wine-quality-dataset")
            return data_asset
        except:
            pass
        
        # Create new data asset
        wine_data = Data(
            path="./wine_quality_complete.csv",
            type=AssetTypes.URI_FILE,
            description="Wine Quality Dataset for Binary Classification",
            name="wine-quality-dataset",
            version="1"
        )
        
        # Register data asset
        data_asset = ml_client.data.create_or_update(wine_data)
        print(f"Data asset created: {data_asset.name}")
        return data_asset
        
    except Exception as e:
        print(f"Error creating data asset: {e}")
        return None

def register_best_model(ml_client):
    """Register the best model"""
    
    try:
        # Register the locally trained model
        best_model_path = "./best_model.pkl"
        if os.path.exists(best_model_path):
            
            model = Model(
                path=best_model_path,
                name="wine-quality-champion",
                description="Best performing wine quality prediction model",
                version="1"
            )
            
            registered_model = ml_client.models.create_or_update(model)
            print(f"Model registered: {registered_model.name}")
            return registered_model
        else:
            print("Best model not found. Run model training first.")
            return None
            
    except Exception as e:
        print(f"Error registering model: {e}")
        return None

def main():
    """Main function to run the complete Azure ML pipeline"""
    
    print("Azure ML Pipeline for Wine Quality Prediction")
    print("=" * 50)
    print("Using compute instance: sara-clases")
    
    # Connect to Azure ML
    ml_client = connect_to_azure_ml()
    if ml_client is None:
        return
    
    # Register model
    print("\nRegistering best model...")
    registered_model = register_best_model(ml_client)
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"AZURE ML PIPELINE SUMMARY")
    print(f"=" * 60)
    print(f"Workspace: sara-ml-workspace")
    print(f"Compute: sara-clases")
    print(f"Model registered: wine-quality-champion")


if __name__ == "__main__":
    main()