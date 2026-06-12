"""
src/preprocessing.py
====================
Reusable preprocessing utilities for the Credit Card Fraud Detection System.
This module is imported by notebooks. Do NOT add experiment logic here —
that belongs in the notebooks. This module provides clean, reusable functions.

Contents:
- load_data()
- encode_categorical()
- split_data()
- scale_features()
- apply_smote()
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple

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

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features and cyclical time features, and drop irrelevant ID column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df = df.copy()
    
    # Drop transaction_id if present
    if 'transaction_id' in df.columns:
        df = df.drop(columns=['transaction_id'])
        print("Dropped column: 'transaction_id'")
        
    # Cyclical hour encoding
    if 'transaction_hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24.0)
        df = df.drop(columns=['transaction_hour'])
        print("Encoded cyclical time column: 'transaction_hour' -> 'hour_sin', 'hour_cos and drop the column transaction_hour'")
        
    # One-hot encode merchant_category if present
    if 'merchant_category' in df.columns:
        df = pd.get_dummies(df, columns=['merchant_category'], dtype=int)
        print("One-hot encoded column: 'merchant_category'")
        
    return df

def split_data(df: pd.DataFrame, target_col: str = 'is_fraud', test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into stratified train, validation, and test sets.
    Preserves target class distribution (1.51% fraud).
    
    Args:
        df (pd.DataFrame): Processed DataFrame.
        target_col (str): Name of target column. Defaults to 'is_fraud'.
        test_size (float): Proportion of dataset for test split. Defaults to 0.15.
        val_size (float): Proportion of dataset for validation split. Defaults to 0.15.
        random_state (int): Random seed. Defaults to 42.
        
    Returns:
        Tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split test set first
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Split validation set from remaining temp data
    val_prop = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_prop, stratify=y_temp, random_state=random_state
    )
    
    print(f"Split completed (stratified):")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, fraud rate={y_train.mean()*100:.2f}%")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}, fraud rate={y_val.mean()*100:.2f}%")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}, fraud rate={y_test.mean()*100:.2f}%")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scale continuous features using StandardScaler fit on X_train only to prevent data leakage.
    Binary and encoded columns are not scaled.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        
    Returns:
        Tuple: Scaled X_train, X_val, X_test DataFrames.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    # Continuous features to scale
    continuous_cols = ['amount', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']
    
    # Ensure they exist in the splits before scaling
    scale_cols = [col for col in continuous_cols if col in X_train.columns]
    
    if scale_cols:
        scaler = StandardScaler()
        # Fit on training set only
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        # Transform validation and test sets
        X_val[scale_cols] = scaler.transform(X_val[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])
        print(f"Scaled continuous columns: {scale_cols}")
        
    return X_train, X_val, X_test

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance the training set (upsample fraud minority class).
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        random_state (int): Random seed. Defaults to 42.
        
    Returns:
        Tuple: Balanced X_train_smote, y_train_smote.
    """
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. Original shape: {X_train.shape}, Balanced shape: {X_train_res.shape}")
    print(f"  Original fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"  Balanced fraud: {y_train_res.sum()} ({y_train_res.mean()*100:.2f}%)")
    return X_train_res, y_train_res
