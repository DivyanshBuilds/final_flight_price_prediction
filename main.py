"""
Main Pipeline
Author: Divyansh
"""

from src.data_ingestion import load_data
from src.data_validation import validate_data
from src.data_transformer import transform_data
from src.model_trainer import train_models


def main():
    """Run the complete ML pipeline"""
    
    print("\n" + "="*60)
    print("FLIGHT PRICE PREDICTION - ML PIPELINE")
    print("="*60)
    
    # Step 1: Data Ingestion
    print("\n[STEP 1/4] DATA INGESTION")
    print("-"*60)
    df = load_data()
    
    # Step 2: Data Validation
    print("\n[STEP 2/4] DATA VALIDATION")
    print("-"*60)
    validate_data(df)
    
    # Step 3: Data Transformation
    print("\n[STEP 3/4] DATA TRANSFORMATION")
    print("-"*60)
    X_train, X_test, y_train, y_test = transform_data(df)
    
    # Step 4: Model Training
    print("\n[STEP 4/4] MODEL TRAINING")
    print("-"*60)
    lr_model, ann_model, metrics = train_models()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE ✓")
    print("="*60)
    print("\n📊 Final Results:")
    print(f"  Linear Regression Test R²: {metrics['linear_regression']['test']['r2']:.4f}")
    print(f"  ANN Test R²: {metrics['ann']['test']['r2']:.4f}")
    print("\n📁 All outputs saved to:")
    print("  - data/processed/")
    print("  - models/")


if __name__ == "__main__":
    main()