import pandas as pd
import numpy as np
import os
import pickle
from wittgenstein import RIPPER
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

DATA_DIR    = 'data/'
RESULTS_DIR = 'outputs/results/'
FIGURES_DIR = 'outputs/figures/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
val   = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Top features from rule analysis in script 10
# These are the features RIPPER actually used most across 25 rules
top_features = ['f41', 'f161', 'f2', 'f3', 'f90',
                'f160', 'f80', 'f132', 'f113', 'f53']

print(f"Using top {len(top_features)} features: {top_features}")

# We will test different feature set sizes
feature_sets = {
    'top_5':  ['f41', 'f161', 'f2', 'f3', 'f90'],
    'top_10': ['f41', 'f161', 'f2', 'f3', 'f90',
               'f160', 'f80', 'f132', 'f113', 'f53'],
    'top_15': ['f41', 'f161', 'f2', 'f3', 'f90',
               'f160', 'f80', 'f132', 'f113', 'f53',
               'f54', 'f55', 'f162', 'f16', 'f4'],
}

# Balance training data
n_illicit = (train['class'] == 1).sum()
n_licit   = (train['class'] == 0).sum()
repeat_times = min(8, round(n_licit / n_illicit))

illicit_train = train[train['class'] == 1]
licit_train   = train[train['class'] == 0]
illicit_rep   = pd.concat([illicit_train] * repeat_times, ignore_index=True)
train_balanced = pd.concat([illicit_rep, licit_train], ignore_index=True)
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

y_train = train_balanced['class']
y_val   = val['class']
y_test  = test['class']

results = []

for set_name, feat_cols in feature_sets.items():
    print(f"\n{'='*55}")
    print(f"FEATURE SET: {set_name} ({len(feat_cols)} features)")

    X_train = train_balanced[feat_cols]
    X_val   = val[feat_cols]
    X_test  = test[feat_cols]

    # Train RIPPER with best config from script 5
    clf = RIPPER(k=2, max_rules=30, max_rule_conds=7, random_state=42)
    clf.fit(X_train, y_train, pos_class=1)

    print(f"Rules generated: {len(clf.ruleset_)}")

    # Evaluate on validation
    y_val_pred  = pd.Series(clf.predict(X_val)).astype(int)
    y_test_pred = pd.Series(clf.predict(X_test)).astype(int)

    val_f1  = round(f1_score(y_val,  y_val_pred,  zero_division=0), 4)
    val_p   = round(precision_score(y_val,  y_val_pred,  zero_division=0), 4)
    val_r   = round(recall_score(y_val,  y_val_pred,  zero_division=0), 4)

    test_f1 = round(f1_score(y_test, y_test_pred, zero_division=0), 4)
    test_p  = round(precision_score(y_test, y_test_pred, zero_division=0), 4)
    test_r  = round(recall_score(y_test, y_test_pred, zero_division=0), 4)
    test_auc = round(roc_auc_score(y_test, y_test_pred), 4)

    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Validation  — Precision: {val_p}  Recall: {val_r}  F1: {val_f1}")
    print(f"Test        — Precision: {test_p}  Recall: {test_r}  F1: {test_f1}  AUC: {test_auc}")
    print(f"Test CM     — TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")

    results.append({
        'feature_set':   set_name,
        'n_features':    len(feat_cols),
        'n_rules':       len(clf.ruleset_),
        'val_f1':        val_f1,
        'val_precision': val_p,
        'val_recall':    val_r,
        'test_f1':       test_f1,
        'test_precision':test_p,
        'test_recall':   test_r,
        'test_auc':      test_auc,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn)
    })

results_df = pd.DataFrame(results)

# Add original RIPPER as reference row
reference = {
    'feature_set':    'original_100',
    'n_features':     100,
    'n_rules':        25,
    'val_f1':         0.8149,
    'val_precision':  0.8158,
    'val_recall':     0.8140,
    'test_f1':        0.0329,
    'test_precision': 0.0306,
    'test_recall':    0.0355,
    'test_auc':       0.5032,
    'TP': 6, 'FP': 190, 'FN': 163
}
results_df = pd.concat(
    [pd.DataFrame([reference]), results_df],
    ignore_index=True
)

print(f"\n{'='*55}")
print("FEATURE-GUIDED RIPPER — SUMMARY")
print(f"{'='*55}")
print(results_df[['feature_set', 'n_features', 'n_rules',
                   'val_f1', 'test_f1', 'test_precision',
                   'test_recall', 'test_auc']].to_string(index=False))

results_df.to_csv(
    os.path.join(RESULTS_DIR, 'feature_guided_ripper.csv'),
    index=False
)
print(f"\nSaved: outputs/results/feature_guided_ripper.csv")
print("\n=== STEP 11 COMPLETE: Feature-Guided RIPPER Done ===")