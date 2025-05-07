import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer


def load_data(file_path): # load using pandas    
    try:
        df = pd.read_csv(file_path)
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found: '{file_path}'")
        return pd.DataFrame()
    
data_file = 'data/synthetic_health_data.csv'
df = load_data(data_file)

df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Sort data by timestamp
df_sorted = df.sort_values('timestamp')

# 2. Set timestamp as index (this allows time-based operations)
df_indexed = df_sorted.set_index('timestamp')

# 3. Calculate rolling mean and standard deviation
#    - First, create a rolling window object based on time:
rolling_window = df_indexed['heart_rate'].rolling(window=300)
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

print(df_result)




#def prepare_data_part2(df_with_features, test_size=0.2, random_state=42):
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
X = df_result[features]

# disease outcome still th y
y = df_result['disease_outcome']

# missing fill
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

print(X_train, X_test, y_train, y_test)
#    return X_train, X_test, y_train, y_test