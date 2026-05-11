import pandas as pd


def validate_data(df):
    """Validate the flight price dataset"""
    
    # Expected schema
    expected_columns = [
        'Unnamed: 0', 'airline', 'flight', 'source_city', 'departure_time',
        'stops', 'arrival_time', 'destination_city', 'class', 'duration',
        'days_left', 'price'
    ]
    
    expected_dtypes = {
        'Unnamed: 0': 'int64',
        'airline': 'object',
        'flight': 'object',
        'source_city': 'object',
        'departure_time': 'object',
        'stops': 'object',
        'arrival_time': 'object',
        'destination_city': 'object',
        'class': 'object',
        'duration': 'float64',
        'days_left': 'int64',
        'price': 'int64'
    }
    
    # Check schema
    print("="*60)
    print("SCHEMA VALIDATION")
    print("="*60)
    
    actual_columns = list(df.columns)
    if actual_columns != expected_columns:
        raise Exception(f"Schema mismatch!\nExpected: {expected_columns}\nGot: {actual_columns}")
    print("✓ Column names match")
    
    # Check data types
    for col, expected_dtype in expected_dtypes.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            raise Exception(f"Data type mismatch for '{col}'!\nExpected: {expected_dtype}, Got: {actual_dtype}")
    print("✓ Data types match")
    
    # Check duplicates
    print("\n" + "="*60)
    print("DUPLICATE CHECK")
    print("="*60)
    
    total_rows = len(df)
    duplicates = df.duplicated().sum()
    duplicate_percentage = (duplicates / total_rows) * 100
    print(f"Total rows: {total_rows}")
    print(f"Duplicate rows: {duplicates} ({duplicate_percentage:.2f}%)")
    
    # Check missing values
    print("\n" + "="*60)
    print("MISSING VALUES CHECK")
    print("="*60)
    
    missing = df.isnull().sum()
    missing_percentage = (missing / total_rows) * 100
    
    for col in df.columns:
        if missing[col] > 0:
            print(f"{col}: {missing[col]} ({missing_percentage[col]:.2f}%)")
    
    if missing.sum() == 0:
        print("✓ No missing values found")
    
    # Check for negative values in numerical columns
    print("\n" + "="*60)
    print("NEGATIVE VALUES CHECK")
    print("="*60)
    
    if (df['duration'] < 0).any():
        print(f"⚠ Warning: Negative values found in 'duration'")
    else:
        print("✓ No negative values in 'duration'")
    
    if (df['days_left'] < 0).any():
        print(f"⚠ Warning: Negative values found in 'days_left'")
    else:
        print("✓ No negative values in 'days_left'")
    
    if (df['price'] < 0).any():
        print(f"⚠ Warning: Negative values found in 'price'")
    else:
        print("✓ No negative values in 'price'")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE ✓")
    print("="*60)
    
    return True


if __name__ == "__main__":
    from data_ingestion import load_data
    
    df = load_data()
    validate_data(df)