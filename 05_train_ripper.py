import pandas as pd
import numpy as np
import os
import json
import pickle
from wittgenstein import RIPPER
from sklearn.metrics import f1_score

# Paths
DATA_DIR = 'data/'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
WEIGHTS_PATH = os.path.join(DATA_DIR, 'class_weights.json')
VAL_PATH = os.path.join(DATA_DIR, 'val.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'ripper_model.pkl')

# Load training data
train = pd.read_csv(TRAIN_PATH)

# Load class weights
with open(WEIGHTS_PATH, 'r') as f:
    class_weights = json.load(f)

weight_illicit = class_weights['1']
weight_licit = class_weights['0']

print(f"Loaded training data: {train.shape}")
print(f"Class weights — Illicit: {weight_illicit}, Licit: {weight_licit}")

# Separate features and labels
feature_cols = [col for col in train.columns if col.startswith('f')]
X_train = train[feature_cols]
y_train = train['class']

print(f"Features: {len(feature_cols)}")
print(f"Illicit transactions: {(y_train == 1).sum()}")
print(f"Licit transactions: {(y_train == 0).sum()}")

# Repeat illicit transactions according to weight ratio
n_illicit = (train['class'] == 1).sum()
n_licit   = (train['class'] == 0).sum()

repeat_times = round(n_licit / n_illicit)
repeat_times = min(repeat_times, 10)

illicit_train = train[train['class'] == 1]
licit_train = train[train['class'] == 0]

illicit_repeated = pd.concat([illicit_train] * repeat_times, ignore_index=True)

train_balanced = pd.concat([illicit_repeated, licit_train], ignore_index=True)
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Update X and y
X_train = train_balanced[feature_cols]
y_train = train_balanced['class']

print(f"\nAfter balancing:")
print(f"Illicit transactions: {(y_train == 1).sum()}")
print(f"Licit transactions: {(y_train == 0).sum()}")

# Load validation set for hyperparameter selection
val = pd.read_csv(VAL_PATH)
X_val = val[feature_cols]
y_val = val['class']

# Hyperparameter tuning — try multiple configs, pick best on validation
print("\nHyperparameter tuning — testing configurations...")

configs = [
    {"max_rules": 10, "max_rule_conds": 3},
    {"max_rules": 20, "max_rule_conds": 5},
    {"max_rules": 25, "max_rule_conds": 5},
    {"max_rules": 30, "max_rule_conds": 7},
]

best_model  = None
best_f1     = 0
best_config = None

for config in configs:
    print(f"\n  Testing config: {config}")

    clf = RIPPER(k=2, random_state=42, **config)
    clf.fit(X_train, y_train, pos_class=1)

    y_val_pred = pd.Series(clf.predict(X_val)).astype(int)
    f1 = round(f1_score(y_val, y_val_pred, zero_division=0), 4)

    print(f"  Rules generated: {len(clf.ruleset_)}")
    print(f"  Validation F1:   {f1}")

    if f1 > best_f1:
        best_f1     = f1
        best_model  = clf
        best_config = config

print(f"\nBest config: {best_config}")
print(f"Best validation F1: {best_f1}")

clf = best_model

# Print winning rules
print(f"\nRIPPER generated {len(clf.ruleset_)} rules")
print("\nRules:")
print(clf.ruleset_)

# Save model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(clf, f)

# Print summary
print("\n=== STEP 5 COMPLETE: RIPPER Training ===")
print(f"Training set size: {len(X_train)}")
print(f"Number of features: {len(feature_cols)}")
print(f"Number of rules generated: {len(clf.ruleset_)}")
print(f"\nModel saved to: {MODEL_PATH}")