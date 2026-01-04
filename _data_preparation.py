# -*- coding: utf-8 -*-
"""
Data Preparation Pipeline for Liver Disease Classification

This module handles data loading, cleaning, imputation, splitting, resampling, and scaling
for supervised learning on the Indian Liver Patient dataset.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTEENN

# ========================================================================
# CONFIGURATION: DATA PATH SETUP
# ========================================================================
# Get the absolute path to the directory where this script is located
# This ensures the script works regardless of where it's executed from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the raw CSV data file
# Uses BASE_DIR to create a robust, cross-platform path
DATA_PATH = os.path.join(BASE_DIR, "data/raw/indian_liver_patient.csv")


# ========================================================================
# STEP 1: LOAD DATA
# ========================================================================
# Load the CSV file into a pandas DataFrame
print("\n" + "="*70)
print("STEP 1: LOADING DATA")
print("="*70)
df = pd.read_csv(DATA_PATH)
print(f"✓ Data loaded successfully from: {DATA_PATH}")
print(f"✓ Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ========================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ========================================================================
print("\n" + "="*70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Data types and info ---")
df.info()

print("\n--- Target variable distribution ---")
print(df['Dataset'].value_counts())
print("\nTarget variable proportions:")
print(df['Dataset'].value_counts(normalize=True))

print("\n--- Missing values count ---")
print(df.isnull().sum())

print("\n--- Missing values percentage ---")
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct)

print("\n--- Basic statistics ---")
print(df.describe())

# ========================================================================
# STEP 3: DATA VISUALIZATION
# ========================================================================
print("\n" + "="*70)
print("STEP 3: DATA VISUALIZATION")
print("="*70)

# Create output directory for saving plots
output_dir = os.path.join(BASE_DIR, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Visualization 1: Histograms of numerical features
# Shows distribution of all numerical columns
# Useful for understanding data spread and identifying skewness
print("\n--- Creating histograms of numerical features ---")
plt.figure(figsize=(15, 10))
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_histograms.png"), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_histograms.png")

# Visualization 2: Gender distribution bar chart
# Shows count of male vs female patients
# Important for understanding demographic composition
print("\n--- Creating gender distribution chart ---")
print(df['Gender'].value_counts())
print("\nGender proportions:")
print(df['Gender'].value_counts(normalize=True))

plt.figure(figsize=(8, 5))
df['Gender'].value_counts().plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Gender Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_gender_distribution.png"), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_gender_distribution.png")

# Visualization 3: Correlation heatmap
# Shows correlation between numerical features
# Red = positive correlation, Blue = negative correlation
# Helps identify feature relationships and potential multicollinearity
print("\n--- Creating correlation heatmap ---")
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True, fmt='.2f', 
            cbar_kws={'label': 'Correlation'}, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_correlation_heatmap.png"), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_correlation_heatmap.png")

# ========================================================================
# STEP 4: DATA CLEANING
# ========================================================================
print("\n" + "="*70)
print("STEP 4: DATA CLEANING")
print("="*70)

# Standardize column names to lowercase
# This ensures consistent column naming and prevents case-sensitivity issues
print("\n--- Standardizing column names to lowercase ---")
df.columns = df.columns.str.strip().str.lower()
print("✓ Column names standardized")
print(f"Column names: {df.columns.tolist()}")

# Encode categorical variables
# Converts categorical text values to numerical representations
print("\n--- Encoding categorical variables ---")
print(f"Original gender values: {df['gender'].unique()}")
df['gender'] = (
    df['gender']
    .str.strip()
    .str.lower()
)
df['gender'] = df['gender'].map({'male': 1, 'female': 0})
print(f"Encoded gender values: {df['gender'].unique()}")
print("\nGender value counts after encoding:")
print(df['gender'].value_counts())

# Fix target labels
# Maps original labels (1, 2) to (1, 0) for binary classification
# 1 = liver disease patient, 0 = non-liver disease patient
print("\n--- Fixing target labels ---")
print(f"Original dataset values: {sorted(df['dataset'].unique())}")
df['dataset'] = df['dataset'].map({1: 1, 2: 0})
print(f"Fixed dataset values: {sorted(df['dataset'].unique())}")
print("\nDataset value counts after fixing:")
print(df['dataset'].value_counts())

print("\n--- Updated data info after cleaning ---")
df.info()

print("✓ Data cleaned successfully!")

# ========================================================================
# STEP 5: DATA IMPUTATION
# ========================================================================
print("\n" + "="*70)
print("STEP 5: DATA IMPUTATION")
print("="*70)

# Handle missing values using statistical methods
# Missing values can bias model training and reduce prediction accuracy
print("\n--- Missing values before imputation ---")
print(df.isnull().sum())

print("\n--- Albumin_and_Globulin_Ratio statistics ---")
print(df['albumin_and_globulin_ratio'].describe())

# Calculate median for imputation
# Median is preferred over mean for skewed clinical data distributions
median_ag_ratio = df['albumin_and_globulin_ratio'].median()
print(f"\nMedian value for imputation: {median_ag_ratio}")

# Perform median imputation
# Replaces missing values with the calculated median
df['albumin_and_globulin_ratio'] = df['albumin_and_globulin_ratio'].fillna(median_ag_ratio)

print("\n--- Missing values after imputation ---")
print(df.isnull().sum())
print("✓ Missing values imputed successfully!")

# ========================================================================
# STEP 6: SEPARATE FEATURES AND TARGET
# ========================================================================
print("\n" + "="*70)
print("STEP 6: SEPARATE FEATURES AND TARGET")
print("="*70)

# Separate independent features (X) from dependent target variable (y)
# X: All columns except 'dataset' (input features)
# y: Only the 'dataset' column (output/target to predict)
X = df.drop(columns=['dataset'])
y = df['dataset']

print(f"\n✓ Features (X) shape: {X.shape}")
print(f"✓ Target (y) shape: {y.shape}")

print("\n--- Target class distribution ---")
print("Target variable value counts:")
print(y.value_counts())
print("\nTarget variable proportions:")
print(y.value_counts(normalize=True))

# ========================================================================
# STEP 7: TRAIN-TEST SPLIT
# ========================================================================
print("\n" + "="*70)
print("STEP 7: STRATIFIED TRAIN-TEST SPLIT")
print("="*70)

# Divide data into training (80%) and testing (20%) sets
# Stratified sampling ensures class distribution is preserved in both sets
# random_state=42 ensures reproducibility across runs
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n✓ Train set size: {X_train.shape[0]}")
print(f"✓ Test set size: {X_test.shape[0]}")

print("\n--- Train set class distribution ---")
print("Class proportions in training set:")
print(y_train.value_counts(normalize=True))

print("\n--- Test set class distribution ---")
print("Class proportions in test set:")
print(y_test.value_counts(normalize=True))

# ========================================================================
# STEP 8: HANDLE CLASS IMBALANCE WITH SMOTE-ENN
# ========================================================================
print("\n" + "="*70)
print("STEP 8: CLASS IMBALANCE HANDLING WITH SMOTE-ENN")
print("="*70)

# Apply SMOTE-ENN to handle class imbalance in training data
# SMOTE: Oversamples minority class by creating synthetic samples
# ENN: Removes noisy samples from majority class
# IMPORTANT: Applied ONLY to training data to prevent data leakage
print("\n--- Applying SMOTE-ENN to training data ---")
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

print("\n--- After SMOTE-ENN resampling ---")
print(f"Original X_train shape: {X_train.shape}")
print(f"Resampled X_train shape: {X_train_resampled.shape}")

print("\nResampled class distribution:")
print("Value counts:")
print(y_train_resampled.value_counts())
print("\nClass proportions:")
print(y_train_resampled.value_counts(normalize=True))

# ========================================================================
# STEP 9: FEATURE SCALING
# ========================================================================
print("\n" + "="*70)
print("STEP 9: FEATURE SCALING WITH ROBUSTSCALER")
print("="*70)

# Normalize numerical features to a consistent scale
# RobustScaler is used instead of StandardScaler for robustness to outliers
# Formula: X_scaled = (X - Q1) / (Q3 - Q1), where Q1/Q3 are quartiles
# Scaler is fitted ONLY on training data to prevent data leakage
print("\n--- Applying RobustScaler ---")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ X_train_scaled shape: {X_train_scaled.shape}")
print(f"✓ X_test_scaled shape: {X_test_scaled.shape}")

# Convert scaled arrays back to DataFrames for consistency
# Preserves column names for better interpretability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\n--- Scaled training data info ---")
print(X_train_scaled.info())

print("\n--- Scaled training data statistics ---")
print(X_train_scaled.describe())

print("\n--- Scaled test data info ---")
print(X_test_scaled.info())

print("\n--- Scaled test data statistics ---")
print(X_test_scaled.describe())

# ========================================================================
# SUMMARY AND COMPLETION
# ========================================================================
print("\n" + "="*70)
print("DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)

print("\n--- Final Data Summary ---")
print(f"Training set (scaled): {X_train_scaled.shape}")
print(f"  └─ Samples: {X_train_scaled.shape[0]}")
print(f"  └─ Features: {X_train_scaled.shape[1]}")
print(f"Test set (scaled): {X_test_scaled.shape}")
print(f"  └─ Samples: {X_test_scaled.shape[0]}")
print(f"  └─ Features: {X_test_scaled.shape[1]}")
print(f"Training labels: {y_train_resampled.shape}")
print(f"Test labels: {y_test.shape}")

print("\n--- Class Distribution (Final) ---")
print("Training set:")
print(y_train_resampled.value_counts())
print("\nTest set:")
print(y_test.value_counts())

print("\n--- Output files saved to ---")
print(f"Visualizations: {output_dir}")
print("\n--- Visualization files ---")
print("  ✓ 01_histograms.png")
print("  ✓ 02_gender_distribution.png")
print("  ✓ 03_correlation_heatmap.png")

print("\n" + "="*70)
print("All data is now ready for machine learning model training!")
print("="*70 + "\n")