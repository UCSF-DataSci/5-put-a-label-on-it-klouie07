## Install necessary packages

# %pip install -r requirements.txt

## 1. Setup

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

## 2. Data Loading

def load_data(file_path): # load using pandas    
    try:
        df = pd.read_csv(file_path)
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found: '{file_path}'")
        return pd.DataFrame()

## 3. Feature Engineering

def extract_rolling_features(df, window_size_seconds):
    """
    Calculate rolling mean and standard deviation for heart rate.
    
    Args:
        df: DataFrame with timestamp and heart_rate columns
        window_size_seconds: Size of the rolling window in seconds
        
    Returns:
        DataFrame with added hr_rolling_mean and hr_rolling_std columns
    """

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. Sort data by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # 2. Set timestamp as index (this allows time-based operations)
    df_indexed = df_sorted.set_index('timestamp')
    
    # 3. Calculate rolling mean and standard deviation
    #    - First, create a rolling window object based on time:
    rolling_window = df_indexed['heart_rate'].rolling(f'{window_size_seconds}s')
    #    - Then calculate statistics on this window:
    hr_mean = rolling_window.mean()
    hr_std = rolling_window.std()
    
    # 4. Add the new columns back to the dataframe
    df_indexed['hr_rolling_mean'] = hr_mean
    df_indexed['hr_rolling_std'] = hr_std
    
    # 5. Reset index to bring timestamp back as a column
    df_result = df_indexed.reset_index()
    
    # 6. Handle any NaN values (rolling calculations create NaNs at the beginning)
    #    - You can use fillna, dropna, or other methods depending on your strategy
    df_result = df_result.bfill()  # Example: backward fill
    
    # Placeholder return - replace with your implementation
    return df_result

## 4. Data Preparation

def prepare_data_part2(df_with_features, test_size=0.2, random_state=42):
    """
    Prepare data for modeling with time-series features.
    
    Args:
        df_with_features: DataFrame with original and rolling features
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # set features
    features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi', 'hr_rolling_mean', 'hr_rolling_std']
    X = df_with_features[features]
    
    # disease outcome still th y
    y = df_with_features['disease_outcome']

    # missing fill
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


## 5. Random Forest Model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Random Forest model
    """
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    rf_model.fit(X_train, y_train)

    return rf_model

## 6. XGBoost Model

def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        learning_rate: Boosting learning rate
        max_depth: Maximum depth of a tree
        random_state: Random seed for reproducibility
        
    Returns:
        Trained XGBoost model
    """
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_label_encoder=False, # kept getting a warning
        random_state=random_state
    )

    xgb_model.fit(X_train, y_train)

    return xgb_model

## 7. Model Comparison

def compare_model_auc(random_forest_model, xgboost_model, X_test, y_test):
    # 1. Generate probability predictions for both models
    rf_probs = random_forest_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgboost_model.predict_proba(X_test)[:, 1]
    
    # 2. Calculate AUC scores
    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    
    # 3. Compare the performance
    return {
        'random_forest_auc': rf_auc,
        'xgboost_auc': xgb_auc
    }


## 8. Save Results
import os

def save_auc_scores(auc_scores, file_path='results/results_part2.txt'):
    """
    Save AUC scores to a text file.
    
    Args:
        auc_scores: Dictionary with AUC scores
        file_path: File path to save the scores
    """
    # 1. Create 'results' directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write("AUC for Part 2 (Random Forest vs XGBoost AUC)\n")
        
        # 2. Format AUC scores as strings
        random_forest_auc_str = f"{auc_scores['random_forest_auc']:.4f}"
        xgboost_auc_str = f"{auc_scores['xgboost_auc']:.4f}"
        
        # 3. Write scores to 'results/results_part2.txt'
        f.write(f"Random Forest AUC: {random_forest_auc_str}\n")
        f.write(f"XGBoost AUC: {xgboost_auc_str}\n")

## 9. Main Execution

# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Extract rolling features
    window_size = 300  # 5 minutes in seconds
    df_with_features = extract_rolling_features(df, window_size)
    
    # 3. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part2(df_with_features, test_size=0.2, random_state=42)
    
    # 4. Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # 5. Calculate AUC scores
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    
    # 6. Save results
    auc_scores = { # dictionary
        'random_forest_auc': rf_auc,
        'xgboost_auc': xgb_auc
    }
    
    save_auc_scores(auc_scores)