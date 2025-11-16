# -*- coding: utf-8 -*-
"""
# Final Project

Name: Aniket Khalate <br>
Professor: Dr.Yasser Abduallah <br>
UCID: ak3274 <br>
Email Id: ak3274@njit.edu <br>

**Objective**: Comparison among several machine learning models that predict diabetes risk using various medical features, and their evaluation through different metrics.

Installation of required packages.<br>
The below line of code installs packages such as pandas, numpy, matplotlib, seaborn, scikit learn and tensorflow.
"""

#pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

"""**Import statements**<br>
The below cell imports the necessary libraries:
1. Basic Data Processing and Analysis: pandas, numpy.
2.  Visualization Libraries: matplotlib, seaborn.
3.  Warning Suppression
4.  Scikit-learn Components: StandardScaler, SVC, RnadomForestClassifier, GridSearchCV, StratifiedKFold, train_test_split, confusion_matrix, roc_auc_score, roc_curve, auc, brier_score_loss.
5.  TensorFlow and Environment Settings: Sequential, Dense, LSTM.
"""


# Import necessary libraries
import warnings
import os
import logging
# Configure warnings and logging to minimize unnecessary output
warnings.filterwarnings("ignore", message="Protobuf gencode version")
logging.getLogger("google.protobuf").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, brier_score_loss
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


"""**Data Loading**: Load the dataset from pima_diabetes.csv for data manipulation and transformation."""

# Load and preprocess data

# Get the folder where this script is located
base_dir = os.getcwd()
diabetes_df = pd.read_csv(os.path.join(base_dir, 'pima_diabetes.csv'))

print("\nDataset Summary:\n")
print(diabetes_df.describe())

print("\nDataset Info:\n")
print(diabetes_df.info())

"""**Data Imputation:** The below function imputes missing data in the diabetes dataset by treating incorrectly recorded zeros. Since measurements like glucose or blood pressure cannot realistically be zero for a living patient, those zeros are identified as missing and replaced with more reasonable estimates."""

def handle_missing_values(diabetes_df):

    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in cols_to_fix:
        diabetes_df.loc[diabetes_df[col] == 0, col] = np.nan
        diabetes_df[col].fillna(diabetes_df[col].median(), inplace=True)

    return diabetes_df

diabetes_df = handle_missing_values(diabetes_df)

"""**Feature and Label Splitting**: Divide the diabetes dataset into two parts<br>

Features (X): The input variables used for prediction


Target (y): The outcome the model is trying to predict


"""

features = diabetes_df.iloc[:, :-1]
target = diabetes_df.iloc[:, -1]

"""**Data Balance Analysis:** The below code checks how the diabetes dataset is distributed by:<br>


1. Counting how many patients have diabetes (positive values)
2. Counting how many do not have diabetes (negative values)
3. Computing the percentage share of each group


"""

positive_values = len(target[target == 1])
negative_values = len(target[target == 0])
total_samples = len(target)

print('\nData Balance Analysis:\n')
print(f'Positive Outcomes: {positive_values} ({(positive_values / total_samples) * 100:.2f}%)')
print(f'Negative Outcomes: {negative_values} ({(negative_values / total_samples) * 100:.2f}%)')

"""**Train Test Split:** The code splits the dataset into training and testing set while also maintaining the distribution of the target variable."""

# train test split and standardization
features_train_all, features_test_all, target_train_all, target_test_all = train_test_split(
    features, target, test_size=0.1, random_state=21, stratify=target)

# Reset indices for the training and testing sets
for dataset in [features_train_all, features_test_all, target_train_all, target_test_all]:
    dataset.reset_index(drop=True, inplace=True)

"""**Feature Standardization:** The below code normalizes the feature values by centering them around the mean and scaling them to have unit variance, which helps machine learning models work more effectively."""

scaler = StandardScaler()

features_train_all_scaled = pd.DataFrame(
    scaler.fit_transform(features_train_all),
    columns=features_train_all.columns
)

features_test_all_scaled = pd.DataFrame(
    scaler.transform(features_test_all),
    columns=features_test_all.columns
)

features_train_all_scaled.describe()

"""**Hyperparameter Tuning:**

1. This step tunes the hyperparameters for the Random Forest and SVM models used in predicting diabetes.

2. The goal is to find the best performing parameter settings while keeping the computation reasonable.

3. GridSearchCV from scikit learn is used to run a thorough search across the chosen parameter grids.


"""

# Grid search for optimal parameters
print("\nUsing grid search for optimal parameters\n")

param_grid_rf = {
    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "min_samples_split": [2, 4, 6, 8, 10]
}

# Grid search for Random Forest
rf_classifier = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=10, n_jobs=-1)
grid_search_rf.fit(features_train_all_scaled, target_train_all)
best_rf_params = grid_search_rf.best_params_
print(f"Best Random Forest parameters: {best_rf_params}")

# Grid search for SVM
param_grid_svc = {"kernel": ["linear"], "C": range(1, 11)}
svc_classifier = SVC(probability=True)
grid_search_svc = GridSearchCV(svc_classifier, param_grid_svc, cv=10, n_jobs=-1)
grid_search_svc.fit(features_train_all_scaled, target_train_all)
best_svc_params = grid_search_svc.best_params_
print(f"Best SVM parameters: {best_svc_params}")

"""**Classification Metrics Calculator:** This function computes key performance metrics using a binary confusion matrix. It is meant for evaluating binary classifiers in machine learning tasks. The resulting metrics like accuracy, precision, recall, and several skill scores that give a overall view of how the model is performing and make it easier to compare different models.


"""

def calculate_performance_metrics(config_matrix):

    TP, FN = config_matrix[0][0], config_matrix[0][1]
    FP, TN = config_matrix[1][0], config_matrix[1][1]

    # basic rates
    TPR = TP / (TP + FN)  # Sensitivity
    TNR = TN / (TN + FP)  # Specificity
    FPR = FP / (TN + FP)  # False Positive Rate
    FNR = FN / (TP + FN)  # False Negative Rate

    # advanced metrics
    Precision = TP / (TP + FP)
    F1_measure = 2 * TP / (2 * TP + FP + FN)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Error_rate = (FP + FN) / (TP + FP + FN + TN)
    BACC = (TPR + TNR) / 2  # Balanced Accuracy

    # skill scores
    TSS = TPR - FPR  # True Skill Statistic
    HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))  # Heidke Skill Score

    return [TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS]

"""**Model Evaluation Function:**

1. This function trains a machine learning model and assesses the performance using a variety of metrics.

2. It works with both traditional ML models and LSTM neural networks, applying the proper preprocessing and evaluation for each.

3. Designed for binary classification, it provides metrics such as confusion matrix scores, ROC AUC, and the Brier score.
"""

def evaluate_model_performance(model, X_train, X_test, y_train, y_test, lstm_flag):

    if lstm_flag:
        # Reshape for LSTM
        X_train_array = X_train.to_numpy()
        X_test_array = X_test.to_numpy()
        X_train_reshaped = X_train_array.reshape(len(X_train_array), X_train_array.shape[1], 1)
        X_test_reshaped = X_test_array.reshape(len(X_test_array), X_test_array.shape[1], 1)

        # Train and evaluate LSTM
        model.fit(X_train_reshaped, y_train, epochs=50,
                  validation_data=(X_test_reshaped, y_test), verbose=0)
        predict_prob = model.predict(X_test_reshaped)
        predicted_labels = (predict_prob > 0.5).astype(int)
        config_matrix = confusion_matrix(y_test, predicted_labels, labels=[1, 0])

        # Calculate metrics for LSTM
        brier_score = brier_score_loss(y_test, predict_prob)
        p = y_test.mean()
        bs_ref = np.mean((p - y_test)**2)
        #Brier Skill Score
        brier_skill_score = 1 - (brier_score / bs_ref)
        roc_auc = roc_auc_score(y_test, predict_prob)
        accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)[1]

    else:
        # Train and evaluate Random Forest & SVM models
        model.fit(X_train, y_train)
        predicted_labels = model.predict(X_test)
        config_matrix = confusion_matrix(y_test, predicted_labels, labels=[1, 0])

        # Calculate metrics for random forest & SVM model
        brier_score = brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])
        p = y_test.mean()
        bs_ref = np.mean((p - y_test)**2)
        #Brier Skill Score
        brier_skill_score = 1 - (brier_score / bs_ref)

        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        accuracy = model.score(X_test, y_test)

    # Combine all metrics
    metrics = calculate_performance_metrics(config_matrix)
    metrics.extend([brier_score, brier_skill_score, roc_auc, accuracy])
    return metrics

"""**Cross Validation Function:**

1. Performs stratified k fold cross validation for multiple models at once.


2. Supports both traditional ML models and LSTM networks, taking care of all required preprocessing and metric calculations.


3. Includes progress tracking, error management, and detailed performance metrics to facilitate model comparison and evaluation.
"""

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)
metrics_dict = {
    'RF': [],
    'SVM': [],
    'LSTM': []
}

# Initialize best_models_dict to track the best performing model for each algorithm
best_models_dict = {
    'RF': None,
    'SVM': None,
    'LSTM': None
}

def run_fold(fold_num, train_idx, test_idx):
    global best_models_dict
    print(f"\nProcessing Fold {fold_num + 1}/10...")

    # Split data for current fold
    X_train = features_train_all_scaled.iloc[train_idx]
    X_test = features_train_all_scaled.iloc[test_idx]
    y_train = target_train_all.iloc[train_idx]
    y_test = target_train_all.iloc[test_idx]

    # Initialize models
    models = {
        'RF': RandomForestClassifier(**best_rf_params),
        'SVM': SVC(**best_svc_params, probability=True),
        'LSTM': Sequential([
            LSTM(64, activation='relu', input_shape=(8, 1), return_sequences=False),
            Dense(1, activation='sigmoid')
        ])
    }

    # Compile LSTM
    models['LSTM'].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train and evaluate each model
    current_fold_metrics = {}
    for name, model in models.items():

        metrics = evaluate_model_performance(
            model, X_train, X_test,
            y_train, y_test,
            name == 'LSTM'
        )
        metrics_dict[name].append(metrics)
        current_fold_metrics[name] = metrics

        # Update best model if accuracy of current fold is better
        if best_models_dict[name] is None or metrics[10] > best_models_dict[name]['accuracy']:
            best_models_dict[name] = {
                'model': model,
                'accuracy': metrics[10]
            }


    metric_columns = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR','Precision', 'F1_measure',
                      'Accuracy', 'Error_rate', 'BACC','TSS', 'HSS', 'Brier_Score', 'Brier_Skill_Score', 'AUC', 'Acc_package_fn']


    df = pd.DataFrame(current_fold_metrics, index=metric_columns)
    print(f"\nFold {fold_num + 1} Results:\n")
    print(df.round(3).to_string())
    print("-" * 70)

    return current_fold_metrics

# Result of each fold
for fold_num, (train_idx, test_idx) in enumerate(cv_strategy.split(features_train_all_scaled, target_train_all)):
    fold_metrics = run_fold(fold_num, train_idx, test_idx)

"""**Average of Metrics:** The below code calculates the average of all folds and gives proper metrics to compare across all models and folds."""

def display_avg_metrics(metrics_dict):

    metric_columns = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR','Precision', 'F1_measure',
                      'Accuracy', 'Error_rate', 'BACC','TSS', 'HSS', 'Brier_Score', 'Brier_Skill_Score', 'AUC', 'Acc_package_fn']

    # Calculate mean metrics
    mean_metrics = {name: np.mean(metrics, axis=0)
                    for name, metrics in metrics_dict.items()}

    metrics_df = pd.DataFrame(mean_metrics, index=metric_columns)

    # Display full metrics table
    print("\nMean Performance Metrics Across All Folds:\n")
    print(metrics_df.round(3).to_string())

display_avg_metrics(metrics_dict)

"""**Evaluating Algorithm Performance:** Compare the ROC curves and AUC scores of different algorithms to assess their performance on the test dataset."""

def plot_roc_curves(X_test_scaled, y_test):

    print("\nPlotting ROC curves")
    colors = {'RF': 'darkorange', 'SVM': 'darkorange', 'LSTM': 'darkorange'}

    for name, model_dict in best_models_dict.items():
        plt.figure(figsize=(8, 8))
        model = model_dict['model']

        if name == 'LSTM':
            X_test_reshaped = X_test_scaled.to_numpy().reshape(-1, 8, 1)
            y_score = model.predict(X_test_reshaped)
        else:
            y_score = model.predict_proba(X_test_scaled)[:, 1]

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc_value = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[name],
                 label=f'ROC curve (AUC = {roc_auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve (Best Model)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

plot_roc_curves(features_test_all_scaled, target_test_all)

def display_avg_metrics(metrics_dict):

    metric_columns = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR','Precision', 'F1_measure',
                      'Accuracy', 'Error_rate', 'BACC','TSS', 'HSS', 'Brier_Score', 'Brier_Skill_Score', 'AUC', 'Acc_package_fn']

    # Calculate mean metrics
    mean_metrics = {name: np.mean(metrics, axis=0)
                    for name, metrics in metrics_dict.items()}

    metrics_df = pd.DataFrame(mean_metrics, index=metric_columns)

    # Summary of metrics
    imp_metrics = ['Accuracy', 'Precision', 'F1_measure', 'AUC', 'BACC']
    sumry_df = metrics_df.loc[imp_metrics]

    print("\nSummary of essential metrics is: \n")
    print(sumry_df.round(3).to_string())

display_avg_metrics(metrics_dict)

"""Performance Comparison and Analysis:
The models were assessed using multiple metrics through 10 fold cross validation.

<br> **1. Random Forest:**
- Achieved high accuracy with well-balanced performance

- Strong ROC-AUC scores

- Computationally efficient

<br> **2. SVM:**

- Delivered competitive accuracy

- Generalized well on test data

- Moderate computational requirements

<br> **3. LSTM:**

- Accuracy comparable to traditional models

- Higher computational demands

- Potential to improve with larger datasets

<br> **4. ROC Curve Analysis:**

- All models showed strong discriminative ability

- ROC curves were clearly separated from the random classifier line

<br> **Conclusion:**
The analysis indicates that all three models perform satisfactorily for diabetes prediction. However, the Random Forest classifier provides the best trade-off between accuracy and computational efficiency. The LSTM, while competitive, demands more resources without offering significant performance gains on this dataset.
"""



