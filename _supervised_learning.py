'''
Our primary goal is Recall >= 0.85 for the disease class
this means : "Catch at least 85% of patients having liver disease"

Application of models are done:

Logistic Regression :simplest of the models , is interpretable ,base benchmark - complex models should pass this benchmark for meaningful output
handles only linear patters.

Random Forest : robust - in the sense of resistant to noise,outliers,overfitting,;handles non linear patters 
if the recalling works well means the class imbalance has worked.

XGBoost : best model and designed for structured tabular data,gives fine grained control over the bias-variance tradeoff.
'''
'''
Decision Metrics:
1.Recall (True Positive Rate)
Formula: TP / (TP + FN)
"Of all the truly diseased patients, what % did we catch?"
Primary metric — must be ≥ 0.85

2.Precision (Positive Predictive Value)
Formula: TP / (TP + FP)
"Of everyone we flagged as diseased, how many actually are?"
Target: ≥ 0.70 (can't be too low or doctors won't trust the model)

3.F1-Score
Formula: 2 * (Precision * Recall) / (Precision + Recall)
"Balanced harmonic mean of precision and recall"
Compare models fairly

4.ROC_AUC
"How good is the model at ranking patients by disease risk?"
Range: 0.5 (random) to 1.0 (perfect)
Target: ≥ 0.80
'''
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data/processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Check all imports
print("All imports successful")

#LOAD DATA:
print("\n"+ "="*70)
print("LOADING PROCESSED DATA FROM CSV DATA FILES")
print("="*70)

#Training data already balanced by SMOTE-ENN
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_processed.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_processed.csv")).squeeze()

#Test data (scaled but NOT balanced — realistic imbalance)
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_processed.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_processed.csv")).squeeze()

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

#check class distribution in training set(after SMOTE ENN ,the classes should be balanced)
print("\nTraining Set Class Distribution->")
print(y_train.value_counts())
print(f"Proportions:\n{y_train.value_counts(normalize=True)}")

#check class distribution in test set
print("\nTest Set Class Distribution->")
print(y_test.value_counts())
print(f"Proportions:\n{y_test.value_counts(normalize=True)}")

#Evaluation of metrics function
def calculate_metrics(y_true,y_pred,y_pred_proba,model_name=""):
    """
    To Calculate comprehensive evaluation metrics for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (0 or 1)
    y_pred : array-like
        Predicted labels (0 or 1) — from model.predict()
    y_pred_proba : array-like
        Predicted probabilities — from model.predict_proba()[:, 1]
    model_name : str
        Name of model (for printing)
    
    Returns:
    --------
    dict with metrics
    """
    recall = recall_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    roc_auc = roc_auc_score(y_true,y_pred_proba)
    
    print(f"\n{model_name} Metrics ->")
    print(f"Recall (Sensitivity):     {recall:.4f}")
    print(f"Precision (PPV):          {precision:.4f}")
    print(f"F1-Score:                 {f1:.4f}")
    print(f"ROC_AUC:                  {roc_auc:.4f}")
    
    return {
        "Model": model_name,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "ROC_AUC": roc_auc
    }
print("--- Evaluation function defined ---")

# TRAIN MODEL 1 - LOGISTIC REGRESSION
print("\n" + "="*70)
print("TRAINING MODEL 1 - LOGISTIC REGRESSION")
print("="*70)

#defining hyperparameter grid
param_grid_lr = {
    'C':[0.001,0.01,0.1,1,10],
    'solver':['lbfgs','liblinear']
}

lr = LogisticRegression(max_iter=1000, random_state=42)

#using GridSearchCV to find best hyperparameters and scoring on f1
print("\nUsing GridSearchCV---")
grid_search_lr = GridSearchCV(
    estimator = lr,
    param_grid = param_grid_lr,
    cv = 5, #5-fold crosvalidation
    scoring = 'f1',
    n_jobs =-1
)

grid_search_lr.fit(X_train,y_train)

print(f"Best parameters: {grid_search_lr.best_params_}")
print(f"Best CV F1 score: {grid_search_lr.best_score_:.4f}")

# Get best model
best_lr = grid_search_lr.best_estimator_

# Make predictions on test set
y_pred_lr = best_lr.predict(X_test)
y_pred_proba_lr = best_lr.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics_lr = calculate_metrics(y_test, y_pred_lr, y_pred_proba_lr, "Logistic Regression")

# TRAIN MODEL 2 - RANDOM FOREST

print("\n" + "="*70)
print("TRAINING MODEL 2 - RANDOM FOREST")
print("="*70)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

print("\nPerforming GridSearchCV...")
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search_rf.fit(X_train, y_train)

print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best CV F1 score: {grid_search_rf.best_score_:.4f}")

best_rf = grid_search_rf.best_estimator_

y_pred_rf = best_rf.predict(X_test)
y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]

metrics_rf = calculate_metrics(y_test, y_pred_rf, y_pred_proba_rf, "Random Forest")


#  TRAIN MODEL 3 - XGBOOST

print("\n" + "="*70)
print("TRAINING MODEL 3 - XGBOOST")
print("="*70)

param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200],
    'subsample': [0.7, 0.9, 1.0]
}

xgb = XGBClassifier(
    random_state=42,
    scale_pos_weight=2,  # Weight disease class more (test is 68% diseased)
    n_jobs=-1
)

print("\nPerforming GridSearchCV...")
grid_search_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_xgb,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search_xgb.fit(X_train, y_train)

print(f"Best parameters: {grid_search_xgb.best_params_}")
print(f"Best CV F1 score: {grid_search_xgb.best_score_:.4f}")

best_xgb = grid_search_xgb.best_estimator_

y_pred_xgb = best_xgb.predict(X_test)
y_pred_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

metrics_xgb = calculate_metrics(y_test, y_pred_xgb, y_pred_proba_xgb, "XGBoost")

#COMPARE MODELS
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

# Create comparison dataframe
comparison_df = pd.DataFrame([metrics_lr, metrics_rf, metrics_xgb])
print("\n--- Model Comparison Table ---")
print(comparison_df.to_string(index=False))

# Find best model based on recall (primary metric)
best_model_idx = comparison_df['Recall'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_recall = comparison_df.loc[best_model_idx, 'Recall']

print(f"\nBest Model (Highest Recall): {best_model_name}")
print(f"Recall: {best_recall:.4f} {'MEETS TARGET (>=0.85)' if best_recall >= 0.85 else 'Below target'}")

# Save comparison table
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print(f"\nComparison table saved to outputs/model_comparison.csv")

# VISUALIZATION: ROC CURVES

print("\n" + "="*70)
print("VISUALIZING ROC CURVES")
print("="*70)

plt.figure(figsize=(10, 7))

# Plot ROC curve for each model
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={metrics_lr["ROC_AUC"]:.3f})', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={metrics_rf["ROC_AUC"]:.3f})', linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={metrics_xgb["ROC_AUC"]:.3f})', linewidth=2)

# Plot random classifier baseline
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC=0.5)')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison - All Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_roc_curves_comparison.png"), dpi=100, bbox_inches='tight')
plt.close()

print("[OK] ROC curve comparison saved to outputs/02_roc_curves_comparison.png")

# Create 3 subplots for individual ROC curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Logistic Regression
axes[0].plot(fpr_lr, tpr_lr, color='blue', linewidth=2, label=f'LR (AUC={metrics_lr["ROC_AUC"]:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
axes[0].fill_between(fpr_lr, tpr_lr, alpha=0.2, color='blue')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Logistic Regression ROC Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Random Forest
axes[1].plot(fpr_rf, tpr_rf, color='green', linewidth=2, label=f'RF (AUC={metrics_rf["ROC_AUC"]:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
axes[1].fill_between(fpr_rf, tpr_rf, alpha=0.2, color='green')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Random Forest ROC Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# XGBoost
axes[2].plot(fpr_xgb, tpr_xgb, color='red', linewidth=2, label=f'XGB (AUC={metrics_xgb["ROC_AUC"]:.3f})')
axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
axes[2].fill_between(fpr_xgb, tpr_xgb, alpha=0.2, color='red')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('XGBoost ROC Curve')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02b_roc_curves_individual.png"), dpi=100, bbox_inches='tight')
plt.close()

print("[OK] Individual ROC curves saved to outputs/02b_roc_curves_individual.png")

# Create confusion matrices for each model
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

# Plot heatmaps
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Logistic Regression\nConfusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
axes[0].set_xticklabels(['Healthy', 'Diseased'])
axes[0].set_yticklabels(['Healthy', 'Diseased'])

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title('Random Forest\nConfusion Matrix')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')
axes[1].set_xticklabels(['Healthy', 'Diseased'])
axes[1].set_yticklabels(['Healthy', 'Diseased'])

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Reds', ax=axes[2], cbar=False)
axes[2].set_title('XGBoost\nConfusion Matrix')
axes[2].set_ylabel('Actual')
axes[2].set_xlabel('Predicted')
axes[2].set_xticklabels(['Healthy', 'Diseased'])
axes[2].set_yticklabels(['Healthy', 'Diseased'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02c_confusion_matrices.png"), dpi=100, bbox_inches='tight')
plt.close()

print("[OK] Confusion matrices saved to outputs/02c_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(10, 7))

# Calculate precision-recall curves
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_proba_lr)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_proba_rf)
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_pred_proba_xgb)

# Plot curves
ax.plot(recall_lr, precision_lr, linewidth=2, label=f'Logistic Regression')
ax.plot(recall_rf, precision_rf, linewidth=2, label=f'Random Forest')
ax.plot(recall_xgb, precision_xgb, linewidth=2, label=f'XGBoost')

# Add baseline (no skill)
ax.axhline(y=y_test.sum() / len(y_test), color='k', linestyle='--', linewidth=1, 
           label=f'Baseline (AUC={y_test.sum() / len(y_test):.2f})')

ax.set_xlabel('Recall (True Positive Rate)', fontsize=12)
ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
ax.set_title('Precision-Recall Curve Comparison\n(Better for Imbalanced Classification)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02d_precision_recall_curves.png"), dpi=100, bbox_inches='tight')
plt.close()

print("[OK] Precision-recall curves saved to outputs/02d_precision_recall_curves.png")

# ========================================================================
# CROSS-VALIDATION ANALYSIS
# ========================================================================
print("\n" + "="*70)
print("VALIDATING MODEL CONSISTENCY ACROSS DATA SPLITS")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models for cross-validation
models_for_cv = {
    'Logistic Regression': LogisticRegression(C=grid_search_lr.best_params_['C'], 
                                               solver=grid_search_lr.best_params_['solver'],
                                               max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=grid_search_rf.best_params_['n_estimators'],
                                           max_depth=grid_search_rf.best_params_['max_depth'],
                                           min_samples_split=grid_search_rf.best_params_['min_samples_split'],
                                           random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(learning_rate=grid_search_xgb.best_params_['learning_rate'],
                            max_depth=grid_search_xgb.best_params_['max_depth'],
                            n_estimators=grid_search_xgb.best_params_['n_estimators'],
                            subsample=grid_search_xgb.best_params_['subsample'],
                            random_state=42, scale_pos_weight=2)
}

cv_results = {}

for model_name, model in models_for_cv.items():
    print(f"\n--- {model_name} ---")
    
    scoring = {
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_scores = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    print(f"\nFold-by-Fold Results:")
    for fold in range(5):
        print(f"  Fold {fold+1}: Recall={cv_scores['test_recall'][fold]:.4f}, " +
              f"Precision={cv_scores['test_precision'][fold]:.4f}, " +
              f"F1={cv_scores['test_f1'][fold]:.4f}, " +
              f"ROC-AUC={cv_scores['test_roc_auc'][fold]:.4f}")
    
    cv_results[model_name] = {
        'Recall': f"{cv_scores['test_recall'].mean():.4f} ± {cv_scores['test_recall'].std():.4f}",
        'Precision': f"{cv_scores['test_precision'].mean():.4f} ± {cv_scores['test_precision'].std():.4f}",
        'F1': f"{cv_scores['test_f1'].mean():.4f} ± {cv_scores['test_f1'].std():.4f}",
        'ROC-AUC': f"{cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}"
    }
    
    print(f"\nAverage Across 5 Folds:")
    print(f"  Recall:    {cv_scores['test_recall'].mean():.4f} ± {cv_scores['test_recall'].std():.4f}")
    print(f"  Precision: {cv_scores['test_precision'].mean():.4f} ± {cv_scores['test_precision'].std():.4f}")
    print(f"  F1:        {cv_scores['test_f1'].mean():.4f} ± {cv_scores['test_f1'].std():.4f}")
    print(f"  ROC-AUC:   {cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}")

cv_df = pd.DataFrame(cv_results).T
print("\n" + "="*70)
print("CROSS-VALIDATION SUMMARY")
print("="*70)
print(cv_df.to_string())

cv_df.to_csv(os.path.join(OUTPUT_DIR, "03_cross_validation_results.csv"))
print(f"\n[OK] Cross-validation results saved to outputs/03_cross_validation_results.csv")

print("\n" + "="*70)
print("MODEL CONSISTENCY INTERPRETATION")
print("="*70)
print("""
Low std (+/- < 0.05):
  [OK] Model is CONSISTENT across all data splits
  [OK] Reliable performance (generalizes well)
  
High std (+/- > 0.08):
  [!] Model is INCONSISTENT
  [!] Performance varies depending on which patients are in test set
  [!] May be overfitting or underfitting
""")

# ============================================================================
# CROSS-VALIDATION COMPARISON PLOT: Test vs CV Performance
# ============================================================================
# Create comparison: Test Set Metrics vs Cross-Validation Metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prepare data for comparison
models_list = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
test_recalls = [metrics_lr['Recall'], metrics_rf['Recall'], metrics_xgb['Recall']]

# Parse CV values (format: "0.8289 ± 0.0240")
recall_strings = cv_df['Recall'].values[:3]
cv_recalls = []
cv_recall_stds = []
for s in recall_strings:
    parts = s.split(' ± ')
    cv_recalls.append(float(parts[0]))
    cv_recall_stds.append(float(parts[1]))

test_roc_aucs = [metrics_lr['ROC_AUC'], metrics_rf['ROC_AUC'], metrics_xgb['ROC_AUC']]

# Parse ROC-AUC values
roc_strings = cv_df['ROC-AUC'].values[:3]
cv_roc_aucs = []
cv_roc_stds = []
for s in roc_strings:
    parts = s.split(' ± ')
    cv_roc_aucs.append(float(parts[0]))
    cv_roc_stds.append(float(parts[1]))

# Plot 1: Recall Comparison
x_pos = np.arange(len(models_list))
width = 0.35

axes[0].bar(x_pos - width/2, test_recalls, width, label='Test Set', color='steelblue', alpha=0.8)
axes[0].bar(x_pos + width/2, cv_recalls, width, label='Cross-Validation', color='coral', alpha=0.8)
axes[0].errorbar(x_pos + width/2, cv_recalls, cv_recall_stds, fmt='none', color='black', capsize=5, capthick=2)

axes[0].set_ylabel('Recall Score', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Models', fontsize=11, fontweight='bold')
axes[0].set_title('Recall: Test Set vs Cross-Validation', fontsize=12, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(models_list)
axes[0].set_ylim([0, 1])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (test, cv) in enumerate(zip(test_recalls, cv_recalls)):
    axes[0].text(i - width/2, test + 0.02, f'{test:.2f}', ha='center', fontsize=9)
    axes[0].text(i + width/2, cv + 0.02, f'{cv:.2f}', ha='center', fontsize=9)

# Plot 2: ROC-AUC Comparison
axes[1].bar(x_pos - width/2, test_roc_aucs, width, label='Test Set', color='steelblue', alpha=0.8)
axes[1].bar(x_pos + width/2, cv_roc_aucs, width, label='Cross-Validation', color='coral', alpha=0.8)
axes[1].errorbar(x_pos + width/2, cv_roc_aucs, cv_roc_stds, fmt='none', color='black', capsize=5, capthick=2)

axes[1].set_ylabel('ROC-AUC Score', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Models', fontsize=11, fontweight='bold')
axes[1].set_title('ROC-AUC: Test Set vs Cross-Validation', fontsize=12, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(models_list)
axes[1].set_ylim([0, 1])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (test, cv) in enumerate(zip(test_roc_aucs, cv_roc_aucs)):
    axes[1].text(i - width/2, test + 0.02, f'{test:.2f}', ha='center', fontsize=9)
    axes[1].text(i + width/2, cv + 0.02, f'{cv:.2f}', ha='center', fontsize=9)

plt.suptitle('Test Set vs Cross-Validation: What We Got vs What the Model Actually Generalizes', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02b_test_vs_cv_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Test vs CV comparison plot saved: 02b_test_vs_cv_comparison.png")
print("\nINTERPRETATION:")
print("  • BLUE bars (Test Set) = Performance on our test data (what we see)")
print("  • CORAL bars (Cross-Validation) = Average performance across 5 folds (what generalizes)")
print("  • ERROR BARS = Variability across folds (smaller = more consistent)")
print("  • Gap between bars = Potential overfitting/underfitting")

# ============================================================================
# STEP 3H: SHAP MODEL EXPLAINABILITY
# ============================================================================
print("\n" + "="*70)
print("STEP 3H: SHAP EXPLAINABILITY ANALYSIS")
print("="*70)

# Initialize SHAP explainer with best XGBoost model
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

print("✓ SHAP explainer initialized for XGBoost model")
print(f"✓ SHAP values computed for {X_test.shape[0]} test samples")

# ============================================================================
# Visualization 1: Feature Importance (Mean Absolute SHAP values)
# ============================================================================
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Feature Importance: Mean |SHAP| Values (XGBoost)", fontsize=14, fontweight='bold')
plt.xlabel("Mean |SHAP| Value", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_shap_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature importance plot saved: 04_shap_feature_importance.png")

# ============================================================================
# Visualization 2: SHAP Summary Plot (Beeswarm)
# ============================================================================
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot: Feature Impact on Model Output", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_shap_summary_beeswarm.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP summary beeswarm plot saved: 05_shap_summary_beeswarm.png")

# ============================================================================
# Visualization 3: SHAP Dependence Plots (Top 4 Features)
# ============================================================================
feature_names = X_test.columns.tolist()
top_features_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-4:][::-1]
top_features = [feature_names[i] for i in top_features_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feature_idx in enumerate(top_features_idx):
    ax = axes[idx]
    shap.dependence_plot(feature_idx, shap_values, X_test, ax=ax, show=False)
    ax.set_title(f"Dependence Plot: {feature_names[feature_idx]}", fontweight='bold')

plt.suptitle("SHAP Dependence Plots: Feature Interactions (Top 4 Features)", 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_shap_dependence_plots.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP dependence plots saved: 06_shap_dependence_plots.png")

# ============================================================================
# Visualization 4: Custom Force Plots for Individual Patients (READABLE)
# ============================================================================
# Select 6 patients: 3 high-risk (diseased) and 3 low-risk (healthy)
diseased_indices = np.where(y_test == 1)[0]
healthy_indices = np.where(y_test == 0)[0]

selected_diseased = diseased_indices[:3]
selected_healthy = healthy_indices[:3]
selected_samples = np.concatenate([selected_diseased, selected_healthy])

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
axes = axes.flatten()

for plot_idx, sample_idx in enumerate(selected_samples):
    ax = axes[plot_idx]
    
    # Get top 8 contributing features (4 positive, 4 negative) for readability
    shap_vals = shap_values[sample_idx]
    feature_names_list = X_test.columns.tolist()
    
    # Sort by absolute SHAP value
    top_indices = np.argsort(np.abs(shap_vals))[-8:][::-1]
    top_features_plot = [feature_names_list[i] for i in top_indices]
    top_shap_vals = shap_vals[top_indices]
    
    # Create horizontal bar chart
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_shap_vals]
    y_pos = np.arange(len(top_features_plot))
    
    ax.barh(y_pos, top_shap_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features_plot, fontsize=10)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (feature, shap_val) in enumerate(zip(top_features_plot, top_shap_vals)):
        ax.text(shap_val + 0.05 if shap_val > 0 else shap_val - 0.05, 
                i, f'{shap_val:.2f}', 
                va='center', ha='left' if shap_val > 0 else 'right', 
                fontsize=9, fontweight='bold')
    
    # Get prediction and actual value
    risk_label = "HIGH RISK (Diseased)" if y_test.iloc[sample_idx] == 1 else "LOW RISK (Healthy)"
    prob = best_xgb.predict_proba(X_test.iloc[sample_idx:sample_idx+1])[0][1]
    
    # Title with key info
    ax.set_title(f"Patient {sample_idx}: {risk_label}\n" + 
                 f"Disease Probability: {prob:.1%} | " +
                 f"Base Model Confidence: {explainer.expected_value:.2f}", 
                 fontweight='bold', fontsize=11, pad=10)
    
    # Add interpretation guide
    ax.text(0.98, 0.02, "Red = Pushes toward Disease | Blue = Pushes toward Healthy", 
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("SHAP Force Plots: Top 8 Biomarker Contributions per Patient\n" +
             "(Top 3 High-Risk Patients vs Top 3 Low-Risk Patients)", 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_shap_force_plots_patients.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP force plots saved: 07_shap_force_plots_patients.png")

# ============================================================================
# SHAP Interpretation Guide
# ============================================================================
print("\n" + "="*70)
print("SHAP VALUES INTERPRETATION GUIDE")
print("="*70)
print("""
FEATURE IMPORTANCE (Bar Plot):
  • Shows which biomarkers have largest impact on model decisions overall
  • Mean |SHAP| value = average impact magnitude across all patients
  • Higher bar = more important for diagnosis

SUMMARY PLOT (Beeswarm):
  • Each dot = one patient's SHAP value for a feature
  • X-axis = SHAP value (+ = pushes towards disease, - = pushes towards healthy)
  • Color = feature value (red = high, blue = low)
  • Clustering = feature interaction patterns

DEPENDENCE PLOTS:
  • X-axis = patient's feature value
  • Y-axis = SHAP contribution of that feature
  • Shows nonlinear relationships and interactions
  • Colored dots = other features influencing the relationship

FORCE PLOTS (Individual Patients):
  • Red bars = biomarkers pushing towards disease prediction
  • Blue bars = biomarkers pushing towards healthy prediction
  • Bar length = strength of push
  • Baseline (gray) = model's average prediction (expected value)
  • Final prediction = baseline + sum of all SHAP contributions

CLINICAL APPLICATION:
  ✓ Use feature importance to focus screening on key biomarkers (AST, Albumin, etc.)
  ✓ Interpret individual force plots to explain specific patient predictions
  ✓ Identify which biomarkers abnormal for each patient
  ✓ Compare high-risk vs low-risk force plots to understand disease patterns
""")

print(f"\n✓ All SHAP visualizations saved to outputs/")
print("✓ Use these plots to explain predictions to clinicians and patients")

# ============================================================================
# STEP 3I: RISK STRATIFICATION
# ============================================================================
print("\n" + "="*70)
print("STEP 3I: PATIENT RISK STRATIFICATION")
print("="*70)

# Get probabilities for all test patients
y_proba = best_xgb.predict_proba(X_test)[:, 1]

# Define risk thresholds
low_threshold = 0.30
high_threshold = 0.70

# Create risk stratification
risk_tiers = []
for prob in y_proba:
    if prob < low_threshold:
        risk_tiers.append("LOW RISK")
    elif prob > high_threshold:
        risk_tiers.append("HIGH RISK")
    else:
        risk_tiers.append("MEDIUM RISK")

# Create patient-level explanations
patient_explanations = []

feature_names = X_test.columns.tolist()

for patient_idx in range(len(X_test)):
    # Get top 5 biomarkers (by absolute SHAP contribution)
    shap_contributions = list(zip(feature_names, shap_values[patient_idx]))
    shap_contributions_sorted = sorted(shap_contributions, key=lambda x: abs(x[1]), reverse=True)
    
    top_5_pushing_disease = [x for x in shap_contributions_sorted[:5] if x[1] > 0]
    top_5_pushing_healthy = [x for x in shap_contributions_sorted if x[1] < 0][:5]
    
    # Get patient's actual values
    patient_values = X_test.iloc[patient_idx]
    
    # Build clinical recommendation
    risk_tier = risk_tiers[patient_idx]
    probability = y_proba[patient_idx]
    
    if risk_tier == "HIGH RISK":
        recommendation = "Immediate clinical evaluation and specialist referral recommended"
        severity = "Critical"
    elif risk_tier == "MEDIUM RISK":
        recommendation = "Schedule follow-up screening within 2-4 weeks"
        severity = "Moderate"
    else:
        recommendation = "Continue routine monitoring; repeat screening in 6-12 months"
        severity = "Low"
    
    # Build biomarker explanation
    biomarker_explanation = "Key abnormal biomarkers: "
    if top_5_pushing_disease:
        biomarker_explanation += ", ".join([f"{f[0]} (SHAP: {f[1]:+.3f})" for f in top_5_pushing_disease])
    else:
        biomarker_explanation = "No significantly abnormal biomarkers detected"
    
    patient_explanations.append({
        "Patient_ID": patient_idx,
        "Risk_Tier": risk_tier,
        "Disease_Probability": f"{probability:.1%}",
        "Severity": severity,
        "Top_Biomarkers_Pushing_Disease": ", ".join([f[0] for f in top_5_pushing_disease[:3]]) if top_5_pushing_disease else "None significant",
        "Clinical_Recommendation": recommendation,
        "Actual_Label": "Disease Present" if y_test.iloc[patient_idx] == 1 else "No Disease"
    })

# Create DataFrame
risk_df = pd.DataFrame(patient_explanations)

# Save to CSV
risk_csv_path = os.path.join(OUTPUT_DIR, "04_patient_risk_stratification.csv")
risk_df.to_csv(risk_csv_path, index=False)

print(f"\n✓ Risk stratification for {len(risk_df)} patients completed")
print(f"✓ Results saved to outputs/04_patient_risk_stratification.csv")

# Print summary statistics
print("\n" + "="*70)
print("RISK STRATIFICATION SUMMARY")
print("="*70)

risk_counts = pd.Series(risk_tiers).value_counts()
print(f"\nRisk Distribution:")
for risk, count in risk_counts.items():
    pct = 100 * count / len(risk_tiers)
    print(f"  {risk:12s}: {count:3d} patients ({pct:5.1f}%)")

# Show accuracy by risk tier
print(f"\nPrediction Accuracy by Risk Tier:")
for risk_tier in ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]:
    mask = np.array(risk_tiers) == risk_tier
    if mask.sum() > 0:
        tier_preds = np.array(risk_tiers)[mask]
        tier_actuals = y_test.values[mask]
        
        # Map risk tiers to predictions
        pred_labels = (np.array(y_proba)[mask] >= 0.5).astype(int)
        accuracy = (pred_labels == tier_actuals).sum() / len(tier_actuals)
        
        diseased_count = (tier_actuals == 1).sum()
        print(f"  {risk_tier:12s}: {accuracy:.1%} accurate | {diseased_count}/{mask.sum()} actually diseased")

# ============================================================================
# VISUALIZATION: Risk Tier Distribution
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Risk distribution
risk_counts_sorted = pd.Series(risk_tiers).value_counts().reindex(["LOW RISK", "MEDIUM RISK", "HIGH RISK"])
colors_risk = ["#2ecc71", "#f39c12", "#e74c3c"]
axes[0].bar(risk_counts_sorted.index, risk_counts_sorted.values, color=colors_risk, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel("Number of Patients", fontsize=11, fontweight='bold')
axes[0].set_title("Patient Risk Distribution", fontsize=12, fontweight='bold')
axes[0].set_ylim([0, max(risk_counts_sorted.values) * 1.2])

# Add value labels
for i, (risk, count) in enumerate(risk_counts_sorted.items()):
    axes[0].text(i, count + 1, str(count), ha='center', fontweight='bold', fontsize=11)

# Plot 2: Risk vs Actual Disease Status
risk_disease_matrix = pd.crosstab(
    pd.Series(risk_tiers, name="Predicted Risk"),
    pd.Series(["Disease" if y == 1 else "Healthy" for y in y_test.values], name="Actual Status")
)

risk_disease_matrix.reindex(["LOW RISK", "MEDIUM RISK", "HIGH RISK"]).plot(
    kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.5
)
axes[1].set_ylabel("Number of Patients", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Predicted Risk Tier", fontsize=11, fontweight='bold')
axes[1].set_title("Risk Tiers vs Actual Disease Status", fontsize=12, fontweight='bold')
axes[1].legend(title="Actual Status", fontsize=10)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_patient_risk_stratification_viz.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Risk stratification visualization saved: 08_patient_risk_stratification_viz.png")

# ============================================================================
# SHOW EXAMPLE PATIENTS FROM EACH RISK TIER
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE PATIENTS FROM EACH RISK TIER")
print("="*70)

for risk_tier in ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]:
    tier_mask = risk_df["Risk_Tier"] == risk_tier
    if tier_mask.any():
        example = risk_df[tier_mask].iloc[0]
        print(f"\n{risk_tier} EXAMPLE:")
        print(f"  Patient ID: {example['Patient_ID']}")
        print(f"  Disease Probability: {example['Disease_Probability']}")
        print(f"  Top Biomarkers: {example['Top_Biomarkers_Pushing_Disease']}")
        print(f"  Actual Status: {example['Actual_Label']}")
        print(f"  Recommendation: {example['Clinical_Recommendation']}")

print("\n" + "="*70)
print("RISK STRATIFICATION COMPLETE")
print("="*70)
print("\nClinical Workflow:")
print("  1. Use probability thresholds to assign risk tiers")
print("  2. Review top biomarkers for each patient")
print("  3. Follow clinical recommendations for each tier")
print("  4. Use SHAP force plots for detailed patient counseling")
print("\n✓ All outputs saved to outputs/ directory")