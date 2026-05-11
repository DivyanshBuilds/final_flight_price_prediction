import pandas as pd
import numpy as np
import pickle
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ANN architecture
HIDDEN_UNITS = [128, 64, 32]
HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "linear"

# ANN training config
LEARNING_RATE = 0.001
LOSS = "mse"
BATCH_SIZE = 256
MAX_EPOCHS = 100
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10


def load_processed_data():
    """Load preprocessed train/test splits"""
    print("Loading preprocessed data...")
    
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model and return metrics"""
    
    # Inverse log transform
    y_true_original = np.expm1(y_true)
    y_pred_original = np.expm1(y_pred)
    
    # Calculate metrics
    r2 = r2_score(y_true_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae = mean_absolute_error(y_true_original, y_pred_original)
    
    print(f"\n{model_name} Evaluation (on original price scale):")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: ₹{rmse:.2f}")
    print(f"  MAE: ₹{mae:.2f}")
    
    return {"r2": r2, "rmse": rmse, "mae": mae}


def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train Linear Regression baseline"""
    print("\n" + "="*60)
    print("TRAINING LINEAR REGRESSION")
    print("="*60)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Predictions
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_pred_train, "Linear Regression (Train)")
    test_metrics = evaluate_model(y_test, y_pred_test, "Linear Regression (Test)")
    
    # Save model
    with open("models/linear_regression.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    print("✓ Saved to models/linear_regression.pkl")
    
    return lr_model, {"train": train_metrics, "test": test_metrics}


def build_ann(input_dim):
    """Build ANN architecture"""
    model = Sequential()
    
    # Hidden layers
    for i, units in enumerate(HIDDEN_UNITS):
        if i == 0:
            model.add(Dense(units, activation=HIDDEN_ACTIVATION, input_dim=input_dim))
        else:
            model.add(Dense(units, activation=HIDDEN_ACTIVATION))
    
    # Output layer
    model.add(Dense(1, activation=OUTPUT_ACTIVATION))
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=['mae']
    )
    
    return model


def train_ann(X_train, y_train, X_test, y_test):
    """Train ANN model"""
    print("\n" + "="*60)
    print("TRAINING ANN")
    print("="*60)
    
    # Build model
    input_dim = X_train.shape[1]
    ann_model = build_ann(input_dim)
    
    print("\nModel Architecture:")
    ann_model.summary()
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print(f"\nTraining for max {MAX_EPOCHS} epochs...")
    history = ann_model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1
    )
    print("✓ Training complete")
    
    # Predictions
    y_pred_train = ann_model.predict(X_train, verbose=0).ravel()
    y_pred_test = ann_model.predict(X_test, verbose=0).ravel()
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_pred_train, "ANN (Train)")
    test_metrics = evaluate_model(y_test, y_pred_test, "ANN (Test)")
    
    # Save model
    ann_model.save("models/ann_model.h5")
    print("✓ Saved to models/ann_model.h5")
    
    # Save training history
    history_dict = {
        "loss": [float(x) for x in history.history['loss']],
        "val_loss": [float(x) for x in history.history['val_loss']],
        "mae": [float(x) for x in history.history['mae']],
        "val_mae": [float(x) for x in history.history['val_mae']]
    }
    
    with open("models/training_history.json", "w") as f:
        json.dump(history_dict, f, indent=4)
    print("✓ Saved training history to models/training_history.json")
    
    return ann_model, {"train": train_metrics, "test": test_metrics}


def train_models():
    """Main training pipeline"""
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train Linear Regression
    lr_model, lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test)
    
    # Train ANN
    ann_model, ann_metrics = train_ann(X_train, y_train, X_test, y_test)
    
    # Save metrics summary
    metrics_summary = {
        "linear_regression": lr_metrics,
        "ann": ann_metrics
    }
    
    with open("models/metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE ✓")
    print("="*60)
    print("\nMetrics Summary:")
    print(json.dumps(metrics_summary, indent=2))
    print("\nAll models and artifacts saved to models/")
    
    return lr_model, ann_model, metrics_summary


if __name__ == "__main__":
    train_models()