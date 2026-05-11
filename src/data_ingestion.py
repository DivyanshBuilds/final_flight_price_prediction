"""
Data Ingestion Module
Author: Divyansh
"""

import pandas as pd


def load_data():
    """Load raw flight price data"""
    df = pd.read_csv("data/raw/clean_dataset.csv")
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape}")
    print(df.head())