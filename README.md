# Predictive Sampling for Credit Card Fraud Detection

This project implements fraud detection for credit card transactions using machine learning and demonstrates various sampling techniques to address class imbalance.

## Table of Contents
- [Overview](#overview)
- [Steps](#steps)
  - [1. Import Libraries](#1-import-libraries)
  - [2. Read Transaction Data](#2-read-transaction-data)
  - [3. Analyze Dataset Features](#3-analyze-dataset-features)
  - [4. Check Target Distribution](#4-check-target-distribution)
  - [5. Data Quality Check](#5-data-quality-check)
  - [6. Split Transaction Types](#6-split-transaction-types)
  - [7. Create Distribution Plot](#7-create-distribution-plot)
  - [8. Balance Dataset](#8-balance-dataset)
  - [9. Combine Balanced Data](#9-combine-balanced-data)
  - [10. Generate Sample Sets](#10-generate-sample-sets)
  - [11. Load ML Libraries](#11-load-ml-libraries)
  - [12. Initialize Models](#12-initialize-models)
  - [13. Prepare Results Storage](#13-prepare-results-storage)
  - [14. Evaluate Model Performance](#14-evaluate-model-performance)
  - [15. Save Final Results](#15-save-final-results)
- [Results](#results)
- [Best Models](#best-models)

---

## Overview

This project demonstrates predictive sampling techniques to detect credit card fraud, addressing class imbalance using methods like SMOTE, and evaluating the performance of various machine learning models on different sample sets.

---

## Steps

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```


### 2. Read Transaction Data
```python
transaction_data = pd.read_csv('Creditcard_data.csv')
```
### 3. Analyze Dataset Features
```python
transaction_data.head()
transaction_data.info()
transaction_data.describe()
transaction_data.head(): # Displays the first few rows of the dataset.
transaction_data.info(): # Shows the structure and summary of the dataset.
transaction_data.describe(): #Provides statistical insights into numerical columns.
```
### 4. Check Target Distribution
```python
fraud_distribution = transaction_data["Class"].value_counts()
print("Transaction Types:")
print(fraud_distribution)# Analyzes the distribution of fraud vs. non-fraud transactions.
```
### 5. Data Quality Check
```python
null_count = transaction_data.isna().sum()
print("Null Values Per Feature:")
print(null_count)#Identifies missing or null values in the dataset.
```
### 6. Split Transaction Types
```python
normal_trans = transaction_data[transaction_data['Class'] == 0]
fraud_trans = transaction_data[transaction_data['Class'] == 1]
print('Normal transactions:', normal_trans.shape)
print('Fraudulent transactions:', fraud_trans.shape)# Separates the dataset into normal and fraudulent transactions.
```
### 7. Create Distribution Plot
```python
plt.figure(figsize=(10, 5))
fraud_distribution.plot(kind='barh', color='lightblue', 
                       title="Transaction Type Distribution")
plt.xlabel("Count")
plt.ylabel("Class") #Visualizes the distribution of transaction types.
```
### 8. Balance Dataset
```python
from imblearn.over_sampling import SMOTE
from collections import Counter

target = transaction_data['Class']
features = transaction_data.drop(['Class'], axis=1)

smote_balancer = SMOTE(random_state=42)
features_balanced, target_balanced = smote_balancer.fit_resample(features, target) #Applies SMOTE to balance the dataset by oversampling the minority class.
```
### 9. Combine Balanced Data
```python
processed_data = pd.concat([
    pd.DataFrame(features_balanced),
    pd.DataFrame(target_balanced, columns=['Class'])
], axis=1)

print("Balanced dataset shape:", processed_data.shape)
print("Class distribution:\n", processed_data['Class'].value_counts())
```
### 10. Generate Sample Sets
```python
from sklearn.model_selection import train_test_split
```
# 1. Random Sample
```python
sample1 = processed_data.sample(n=int(0.2 * len(processed_data)), random_state=42)
```
# 2. Stratified Sample
```python
grouped = processed_data.groupby('Class')
sample2 = grouped.apply(lambda x: x.sample(int(0.2 * len(x)), random_state=42)).reset_index(drop=True)
```
# 3. Systematic Sample
```python
interval = len(processed_data) // int(0.2 * len(processed_data))
offset = np.random.randint(0, interval)
sample3 = processed_data.iloc[offset::interval]
```

# 4. Cluster Sample
```python
n_groups = 5
group_ids = np.arange(len(processed_data)) % n_groups
processed_data['Group'] = group_ids
selected_group = np.random.randint(0, n_groups)
sample4 = processed_data[processed_data['Group'] == selected_group].drop('Group', axis=1)
```
# 5. Bootstrap Sample
```python
sample5 = processed_data.sample(n=int(0.2 * len(processed_data)), replace=True, random_state=42)

print("Sample sizes:", len(sample1), len(sample2), len(sample3), len(sample4), len(sample5))
```
# 11. Load ML Libraries
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```
# 12. Initialize Models
```python
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier()
}
```
# 13. Prepare Results Storage
```python
performance_metrics = {}
sample_set = [sample1, sample2, sample3, sample4, sample5]
```
# 14. Evaluate Model Performance
```python
for model_name, classifier in classifiers.items():
    performance_metrics[model_name] = []
    
    for i, sample in enumerate(sample_set):
        X = sample.drop('Class', axis=1)
        y = sample['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        performance_metrics[model_name].append(accuracy)

results_table = pd.DataFrame(performance_metrics, index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"])
print(results_table)
results_table.to_csv("model_accuracy.csv")
```
# 15. Save Final Results
```python
results_table.to_csv('Submission_102203509_ADRIJA.csv')
```
# Results

The model accuracy for each classifier and sampling method is stored in model_accuracy.csv.

## Best Models
- Sample 1: Gradient Boosting
- Sample 2: Logistic Regression
- Sample 3: Decision Tree / Gradient Boosting
- Sample 4: Logistic Regression
- Sample 5: Decision Tree / Gradient Boosting

# Result's Table
| Sample  | Logistic Regression | Decision Tree | Gradient Boosting | SVM       | k-NN      |
|---------|----------------------|---------------|--------------------|-----------|-----------|
| Sample1 | 0.9180327868852459  | 0.9016393442622951 | 0.9344262295081968 | 0.7213114754098361 | 0.6557377049180327 |
| Sample2 | 0.9508196721311475  | 0.9016393442622951 | 0.9344262295081968 | 0.7540983606557377 | 0.7540983606557377 |
| Sample3 | 0.7419354838709677  | 0.9032258064516129 | 0.9516129032258065 | 0.6774193548387096 | 0.7258064516129032 |
| Sample4 | 0.9836065573770492  | 0.9508196721311475 | 0.9508196721311475 | 0.6885245901639344 | 0.7540983606557377 |
| Sample5 | 0.9508196721311475  | 0.9672131147540983 | 0.9672131147540983 | 0.639344262295082  | 0.7540983606557377 |

