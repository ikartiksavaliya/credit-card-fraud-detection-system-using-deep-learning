"""
src/preprocessing.py
====================
Reusable preprocessing utilities for the Credit Card Fraud Detection System.
This module is imported by notebooks. Do NOT add experiment logic here —
that belongs in the notebooks. This module provides clean, reusable functions.

Contents (to be filled as notebooks progress):
- load_data()
- encode_categorical()
- scale_features()
- split_data()
- apply_smote()
"""

import os
import pandas as pd

def load_data(filepath: str = "data/credit_card_fraud_10k.csv") -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    
    Args:
        filepath (str): Path to the CSV file. Defaults to 'data/credit_card_fraud_10k.csv'.
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(filepath):
        # Fallback for running from different subdirectories
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_path = os.path.join(base_dir, filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath} or {alt_path}")
            
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

def encode_categorical(df: pd.DataFrame):
    """
    Encode categorical features.
    To be fully implemented in Phase 3.
    """
    raise NotImplementedError("encode_categorical will be implemented in Phase 3 (Notebook 02).")

def scale_features(X_train, X_val, X_test):
    """
    Scale numeric features.
    To be fully implemented in Phase 3.
    """
    raise NotImplementedError("scale_features will be implemented in Phase 3 (Notebook 02).")

def split_data(df: pd.DataFrame):
    """
    Split the dataset into stratified train, validation, and test sets.
    To be fully implemented in Phase 3.
    """
    raise NotImplementedError("split_data will be implemented in Phase 3 (Notebook 02).")

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance the training set.
    To be fully implemented in Phase 3.
    """
    raise NotImplementedError("apply_smote will be implemented in Phase 3 (Notebook 02).")
