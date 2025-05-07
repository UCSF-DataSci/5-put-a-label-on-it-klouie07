## Install Necessary Packages

# %pip install -r requirements.txt

## 1. Setup

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer


## 2. Data Loading
def load_data(file_path): # load data using pandas
    
    try:
        df = pd.read_csv(file_path)
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found: '{file_path}'")
        return pd.DataFrame()


## 3. Data Preparation
def prepare_data_part1(df, test_size = 0.2, random_state = 42): # use template and select features, split into train/test, handle missing data
    
    features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi']
    target = 'disease_outcome'
    
    X = df[features]
    y = df[target]
    
    imputer = SimpleImputer(strategy = 'mean')
    X_imputed = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size = test_size, random_state = random_state, stratify = y)
    
    return X_train, X_test, y_train, y_test


## 4. Model Training
def train_logistic_regression(X_train, y_train):
    
    model = LogisticRegression(max_iter = 500)  
    model.fit(X_train, y_train)
    
    return model

## 5. Model Evaluation
def calculate_evaluation_metrics(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # setting up the probability of a positive result to compare AUC score
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': conf_matrix
    }
    
    return metrics

## 6. Save Results
import os

def save_results(metrics, file_path = 'results/results_part1.txt'):
    os.makedirs(os.path.dirname(file_path), exist_ok = True) # make directory if no exist
    
    with open(file_path, 'w') as f: 
        f.write("Model Evaluation Metrics\n")
        for metric, value in metrics.items():
            if metric == 'confusion_matrix':
                f.write(f"{metric}:\n{value}\n") # make the thing 2x2
            else:
                f.write(f"{metric}: {value:.4f}\n")

## 8. Interpret Results
def interpret_results(metrics):
    
    # set the mini stuff
    best_metric = None
    worst_metric = None
    imbalance_impact_score = 0
    
    # use the metrics set up from before
    f1 = metrics['f1']
    precision = metrics['precision']
    recall = metrics['recall']
    auc = metrics['auc']
    
    # compare the f1 and AUC to see which is higher 
    # f1 is looking at precision and recall
    # AUC for identification of correct positives and negatives
    best_metric_value = max(f1, auc)
    if best_metric_value == f1:
        best_metric = 'f1'
    else:
        best_metric = 'auc'
    
    # assuming imbalanced set, accuracy doesnt really tell us a lot of info from the data other than how much is right and wrong
    accuracy = metrics['accuracy']    
    worst_metric = 'accuracy'
    
    # if recall and precision are the same, then there's likely no imbalance
    # if they're different, then it is possible there is some bias or imbalance
    imbalance_impact_score = abs(recall - precision)
    
    # set in dictionary
    return {
        'best_metric': best_metric,
        'worst_metric': worst_metric,
        'imbalance_impact_score': imbalance_impact_score
    }

## 7. Main Execution
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part1(df)
    
    # 3. Train model
    model = train_logistic_regression(X_train, y_train)
    
    # 4. Evaluate model
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 5. Print metrics
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # 6. Save results
    save_results(metrics, 'results/results_part1.txt')
    
    # 7. Interpret results
    interpretation = interpret_results(metrics)
    print("\nResults Interpretation:")
    for key, value in interpretation.items():
        print(f"{key}: {value}")
