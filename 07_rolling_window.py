import pandas as pd
import numpy as np
import os
import json
import pickle
from wittgenstein import RIPPER

# Paths
DATA_DIR = 'data/'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
VAL_PATH = os.path.join(DATA_DIR, 'val.csv')
WEIGHTS_PATH = os.path.join(DATA_DIR, 'class_weights.json')

# Window sizes to try
WINDOW_SIZES = [10, 15, 20]

# Load data
train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

# Load class weights
with open(WEIGHTS_PATH, 'r') as f:
    class_weights = json.load(f)

weight_illicit = class_weights['1']
weight_licit = class_weights['0']

# Feature columns
feature_cols = [col for col in train.columns if col.startswith('f')]

# Fixed test set — never changes across experiments
X_test = test[feature_cols]
y_test = test['class']

X_val = val[feature_cols]
y_val = val['class']

print(f"Full training set: {len(train)} transactions")
print(f"Time steps available in training: {sorted(train['time_step'].unique())}")

# Store results for each window size
results = []

for window_size in WINDOW_SIZES:
    print(f"\n{'='*50}")
    print(f"WINDOW SIZE: {window_size} time steps")
    print(f"Training on steps {35 - window_size} to 34")
    
    # Filter training data to most recent window_size steps
    min_step = 35 - window_size
    window_train = train[train['time_step'] >= min_step].copy()
    
    print(f"Transactions in window: {len(window_train)}")
    print(f"Illicit: {(window_train['class']==1).sum()}")
    print(f"Licit: {(window_train['class']==0).sum()}")
    
    # Balance using repetition
    repeat_times = round(weight_illicit / weight_licit)
    
    illicit_window = window_train[window_train['class'] == 1]
    licit_window = window_train[window_train['class'] == 0]
    
    illicit_repeated = pd.concat([illicit_window] * repeat_times, ignore_index=True)
    
    window_balanced = pd.concat([illicit_repeated, licit_window], ignore_index=True)
    window_balanced = window_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_window = window_balanced[feature_cols]
    y_window = window_balanced['class']
    
    print(f"After balancing — Illicit: {(y_window==1).sum()} Licit: {(y_window==0).sum()}")
    
    # Train RIPPER
    clf = RIPPER(
        k=2,
        max_rules=25,
        max_rule_conds=5,
        random_state=42
    )
    
    clf.fit(X_window, y_window, pos_class=1)
    print(f"Rules generated: {len(clf.ruleset_)}")
    
    # Predict on validation and test
    y_val_pred = pd.Series(clf.predict(X_val)).astype(int)
    y_test_pred = pd.Series(clf.predict(X_test)).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    
    val_f1 = round(f1_score(y_val, y_val_pred, zero_division=0), 4)
    val_precision = round(precision_score(y_val, y_val_pred, zero_division=0), 4)
    val_recall = round(recall_score(y_val, y_val_pred, zero_division=0), 4)
    
    test_f1 = round(f1_score(y_test, y_test_pred, zero_division=0), 4)
    test_precision = round(precision_score(y_test, y_test_pred, zero_division=0), 4)
    test_recall = round(recall_score(y_test, y_test_pred, zero_division=0), 4)
    test_auc = round(roc_auc_score(y_test, y_test_pred), 4)
    
    print(f"\nValidation — Precision: {val_precision} Recall: {val_recall} F1: {val_f1}")
    print(f"Test — Precision: {test_precision} Recall: {test_recall} F1: {test_f1} AUC: {test_auc}")
    
    # Store results
    results.append({
        'window_size': window_size,
        'steps': f"{min_step}-34",
        'n_transactions': len(window_train),
        'n_rules': len(clf.ruleset_),
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_auc': test_auc
    })

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Print comparison table
print(f"\n{'='*50}")
print("ROLLING WINDOW COMPARISON")
print(f"{'='*50}")
print(results_df[['window_size', 'steps', 'n_transactions', 'n_rules', 
                   'val_f1', 'test_f1', 'test_precision', 'test_recall', 'test_auc']].to_string(index=False))

# Find best window based on test F1
best = results_df.loc[results_df['val_f1'].idxmax()]

print(f"\n{'='*50}")
print(f"BEST WINDOW SIZE: {int(best['window_size'])} time steps")
print(f"Steps: {best['steps']}")
print(f"Transactions: {int(best['n_transactions'])}")
print(f"Rules: {int(best['n_rules'])}")
print(f"Validation F1: {best['val_f1']}")
print(f"Test F1: {best['test_f1']}")
print(f"Test Precision: {best['test_precision']}")
print(f"Test Recall: {best['test_recall']}")
print(f"Test AUC-ROC: {best['test_auc']}")

# Save results table
results_df.to_csv(os.path.join(DATA_DIR, 'rolling_window_results.csv'), index=False)
print(f"\nResults saved to: data/rolling_window_results.csv")

print("\n=== STEP 7 COMPLETE: Rolling Window Done ===")