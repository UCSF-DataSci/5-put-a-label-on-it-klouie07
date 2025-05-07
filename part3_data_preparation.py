# Install necessary packages

# %pip install -r requirements.txt

## 1. Setup

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


## 2. Data Loading

def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    try:
        df = pd.read_csv(file_path)
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found: '{file_path}'")
        return pd.DataFrame()


## 3. Categorical Feature Encoding

def encode_categorical_features(df, column_to_encode = 'smoker_status'):
    """
    Encode a categorical column using OneHotEncoder.

    Args:
        df: Input DataFrame
        column_to_encode: Name of the categorical column to encode

    Returns:
        DataFrame with the categorical column replaced by one-hot encoded columns
    """
    values = df[[column_to_encode]] # separate column

    encoder = OneHotEncoder(drop = 'first', sparse_output = False) # more than one smoker status? drop = 'first'
    encoded_array = encoder.fit_transform(df[[column_to_encode]])
    encoded_columns = encoder.get_feature_names_out([column_to_encode])
    
    encoded_df = pd.DataFrame(encoded_array, columns = encoded_columns, index = df.index) # redo the values based on data
    
    df_encoded = df.drop(columns = [column_to_encode]) # drop original column
    df_encoded = pd.concat([df_encoded, encoded_df], axis = 1) # add new column
    
    return df_encoded


## 4. Data Preparation

def prepare_data_part3(df, test_size=0.2, random_state=42):
    """
    Prepare data with categorical encoding.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 1. Encode categorical features using the encode_categorical_features function
    df_encoded = encode_categorical_features(df, column_to_encode='smoker_status')
    
    # 2. Select relevant features (including the one-hot encoded ones) and the target
    features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi'] + \
               [col for col in df_encoded.columns if col.startswith('smoker_status_')]
    # encoded columns appear with 'smoker_status_YADA_YADA_HERE' withe the text listed
    X = df_encoded[features]
    y = df_encoded['disease_outcome']
    
    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
    
    # 4. Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    return X_train, X_test, y_train, y_test


## 5. Handling Imbalanced Data

def apply_smote(X_train, y_train, random_state = 42):
    """
    Apply SMOTE to oversample the minority class.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        
    Returns:
        Resampled X_train and y_train with balanced classes
    """
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=random_state)
    
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Placeholder return - replace with your implementation
    return X_train_smote, y_train_smote


## 6. Model Training and Evaluation

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained logistic regression model
    """
    # Initialize and train a LogisticRegression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    return model  # Replace with actual implementation

def calculate_evaluation_metrics(model, X_test, y_test):
    """
    Calculate classification evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc, and confusion_matrix
    """
    # 1. Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability calc for positive class

    # 2. Calculate metrics: accuracy, precision, recall, f1, auc
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # 3. Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 4. Return metrics in a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

## 7. Save Results
import os

def save_results(metrics, file_path):
    # 1. Create 'results' directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok = True) 
    
    with open(file_path, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        for metric, value in metrics.items():
            if metric == 'confusion_matrix':
                f.write(f"{metric}:\n{value}\n")
            #if isinstance(value, (list, tuple)):  # For confusion_matrix
            #    f.write(f"{metric}:\n")
            #    for row in value:
            #        f.write(" ".join(str(x) for x in row) + "\n")
            else:
                f.write(f"{metric}: {value:.4f}\n")


## 9. Compare Results

# added a new definition stuff
def parse_metrics_from_file(file_path): # help with pulling the actual data from the file and ignoring all the other stuff
    metrics = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Loop through each line in the file
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespaces
            
            if line.startswith('accuracy:'):
                metrics['accuracy'] = float(line.split(':')[1].strip())
            elif line.startswith('precision:'):
                metrics['precision'] = float(line.split(':')[1].strip())
            elif line.startswith('recall:'):
                metrics['recall'] = float(line.split(':')[1].strip())
            elif line.startswith('f1:'):
                metrics['f1'] = float(line.split(':')[1].strip())
            elif line.startswith('auc:'):
                metrics['auc'] = float(line.split(':')[1].strip())

    return metrics

def compare_models(part1_metrics, part3_metrics):
    """
    Calculate percentage improvement between models trained on imbalanced vs. balanced data.
    
    Args:
        part1_metrics: Dictionary containing evaluation metrics from Part 1 (imbalanced)
        part3_metrics: Dictionary containing evaluation metrics from Part 3 (balanced)
        
    Returns:
        Dictionary with metric names as keys and improvement percentages as values
    """
    
    improvements = {}

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']: # 2. Handle metrics where higher is better (most metrics) and where lower is better
        # assumption here is that higher accuracy, precision, recall, f1, and AUC are good
        value1 = part1_metrics.get(metric, 0)
        value3 = part3_metrics.get(metric, 0)
        
        if value1 == 0:
            # Avoid division by zero; assume full improvement if value3 > 0
            improvements[metric] = float('inf') if value3 > 0 else 0.0
        else:
            # 1. Calculate percentage improvement for each metric
            improvement = ((value3 - value1) / abs(value1)) * 100 
            # 3. Return a dictionary with metric names and improvement percentages
            improvements[metric] = round(improvement, 2) # rounded to 2 decimals

    return improvements


## 8. Main Execution

# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data with categorical encoding
    X_train, X_test, y_train, y_test = prepare_data_part3(df)
    
    # 3. Apply SMOTE to balance the training data
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 4. Train model on resampled data
    model = train_logistic_regression(X_train_resampled, y_train_resampled)
    
    # 5. Evaluate on original test set
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 6. Print metrics
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # 7. Save results
    save_results(metrics, 'results/results_part3.txt')
    
    # 8. Load Part 1 results for comparison    
    import json
    try:
        #with open('results/results_part1.txt', 'r') as f:
        #    part1_metrics = json.load(f)
        part1_metrics = parse_metrics_from_file('results/results_part1.txt')
        part3_metrics = parse_metrics_from_file('results/results_part3.txt')

        comparison = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        # 9. Compare models
        improvements = compare_models(part1_metrics, part3_metrics) # edited
        print("\nModel Comparison (improvement percentages):") # edited
        print("(Pt3 - Pt1) / |Pt1|") # edited
        print(improvements) # edited
        #for metric, improvement in comparison:
        #    print(f"{metric}: {improvement:.2f}%")
    except FileNotFoundError:
        print("Part 1 results not found. Run part1_introduction.ipynb first.")


