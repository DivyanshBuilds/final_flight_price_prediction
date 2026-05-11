import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import pickle
import os

# Configuration
TARGET_COLUMN = "price"
COLUMNS_TO_DROP = ["Unnamed: 0", "flight"]

# Encoding groups
ONEHOT_COLUMNS = [
    "airline",
    "source_city",
    "departure_time",
    "arrival_time",
    "destination_city",
]
ORDINAL_STOPS_ORDER = [["zero", "one", "two_or_more"]]
ORDINAL_CLASS_ORDER = [["Economy", "Business"]]

# Scaling group
NUMERICAL_COLUMNS = ["duration", "days_left"]

# Split config
TEST_SIZE = 0.2
RANDOM_STATE = 42


def transform_data(df):
    """Transform raw data into model-ready format"""
    
    print("="*60)
    print("DATA TRANSFORMATION PIPELINE")
    print("="*60)
    
    # Step 1: Clean data
    print("\n1. Cleaning data...")
    df = df.drop(columns=COLUMNS_TO_DROP)
    print(f"   Dropped columns: {COLUMNS_TO_DROP}")
    
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"   Removed {duplicates_removed} duplicate rows")
    print(f"   Final shape: {df.shape}")
    
    # Step 2: Separate X and y
    print("\n2. Separating features and target...")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # Step 3: Train-test split
    print("\n3. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Step 4: Log-transform target (AFTER split)
    print("\n4. Log-transforming target variable...")
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)
    print(f"   Applied log1p transformation")
    
    # Step 5: Encode categorical features
    print("\n5. Encoding categorical features...")
    
    # OneHot Encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(X_train[ONEHOT_COLUMNS])
    
    ohe_train = ohe.transform(X_train[ONEHOT_COLUMNS])
    ohe_test = ohe.transform(X_test[ONEHOT_COLUMNS])
    
    ohe_train_df = pd.DataFrame(ohe_train, columns=ohe.get_feature_names_out(), index=X_train.index)
    ohe_test_df = pd.DataFrame(ohe_test, columns=ohe.get_feature_names_out(), index=X_test.index)
    print(f"   OneHot encoded: {ONEHOT_COLUMNS}")
    
    # Ordinal Encoding - stops
    ordinal_stops = OrdinalEncoder(categories=ORDINAL_STOPS_ORDER)
    ordinal_stops.fit(X_train[['stops']])
    
    X_train['stops_encoded'] = ordinal_stops.transform(X_train[['stops']])
    X_test['stops_encoded'] = ordinal_stops.transform(X_test[['stops']])
    print(f"   Ordinal encoded 'stops': {ORDINAL_STOPS_ORDER[0]}")
    
    # Ordinal Encoding - class
    ordinal_class = OrdinalEncoder(categories=ORDINAL_CLASS_ORDER)
    ordinal_class.fit(X_train[['class']])
    
    X_train['class_encoded'] = ordinal_class.transform(X_train[['class']])
    X_test['class_encoded'] = ordinal_class.transform(X_test[['class']])
    print(f"   Ordinal encoded 'class': {ORDINAL_CLASS_ORDER[0]}")
    
    # Step 6: Scale numerical features
    print("\n6. Scaling numerical features...")
    scaler = StandardScaler()
    scaler.fit(X_train[NUMERICAL_COLUMNS])
    
    X_train[NUMERICAL_COLUMNS] = scaler.transform(X_train[NUMERICAL_COLUMNS])
    X_test[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])
    print(f"   Scaled: {NUMERICAL_COLUMNS}")
    
    # Step 7: Combine all features
    print("\n7. Combining encoded features...")
    X_train_final = pd.concat([
        X_train[NUMERICAL_COLUMNS],
        X_train[['stops_encoded', 'class_encoded']],
        ohe_train_df
    ], axis=1)
    
    X_test_final = pd.concat([
        X_test[NUMERICAL_COLUMNS],
        X_test[['stops_encoded', 'class_encoded']],
        ohe_test_df
    ], axis=1)
    
    print(f"   Final train shape: {X_train_final.shape}")
    print(f"   Final test shape: {X_test_final.shape}")
    
    # Step 8: Save processed data
    print("\n8. Saving processed data...")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    X_train_final.to_csv("data/processed/X_train.csv", index=False)
    X_test_final.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False, header=['price'])
    y_test.to_csv("data/processed/y_test.csv", index=False, header=['price'])
    print("   ✓ Saved CSVs to data/processed/")
    
    # Save encoders and scaler
    with open("models/ohe.pkl", "wb") as f:
        pickle.dump(ohe, f)
    with open("models/ordinal_encoder_stops.pkl", "wb") as f:
        pickle.dump(ordinal_stops, f)
    with open("models/ordinal_encoder_class.pkl", "wb") as f:
        pickle.dump(ordinal_class, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("   ✓ Saved encoders and scaler to models/")
    
    print("\n" + "="*60)
    print("TRANSFORMATION COMPLETE ✓")
    print("="*60)
    
    return X_train_final, X_test_final, y_train, y_test


if __name__ == "__main__":
    from data_ingestion import load_data
    
    df = load_data()
    X_train, X_test, y_train, y_test = transform_data(df)
    
    print(f"\nFinal datasets ready for modeling!")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")