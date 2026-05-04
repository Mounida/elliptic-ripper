# 12_ripper_tw.py
# =============================================================================
# RIPPER-TW Experiment: Time-Weighted FOIL Gain
# =============================================================================
# Tests RIPPER-TW (internal time-weighted modification) against standard
# RIPPER across multiple decay values. This is the core contribution script.
# =============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Import our modified algorithm
from ripper_tw import RIPPER_TW
from wittgenstein import RIPPER

DATA_DIR    = 'data/'
RESULTS_DIR = 'outputs/results/'
FIGURES_DIR = 'outputs/figures/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
val   = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

feature_cols = [col for col in train.columns if col.startswith('f')]

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"Features: {len(feature_cols)}")

# ─────────────────────────────────────────────
# Balance training data (same as script 05)
# ─────────────────────────────────────────────
n_illicit    = (train['class'] == 1).sum()
n_licit      = (train['class'] == 0).sum()
repeat_times = min(8, round(n_licit / n_illicit))

illicit_train  = train[train['class'] == 1]
licit_train    = train[train['class'] == 0]
illicit_rep    = pd.concat([illicit_train] * repeat_times, ignore_index=True)
train_balanced = pd.concat([illicit_rep, licit_train], ignore_index=True)
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_train      = train_balanced[feature_cols]
y_train      = train_balanced['class']
time_steps   = train_balanced['time_step']   # ← passed to RIPPER-TW

X_val        = val[feature_cols]
y_val        = val['class']

X_test       = test[feature_cols]
y_test       = test['class']

print(f"\nBalanced train — Illicit: {(y_train==1).sum()} | Licit: {(y_train==0).sum()}")

# ─────────────────────────────────────────────
# Helper: evaluate a fitted model
# ─────────────────────────────────────────────
def evaluate(clf, X, y, label):
    y_pred = pd.Series(clf.predict(X)).astype(int)
    has_both = len(y.unique()) == 2
    return {
        'model':     label,
        'precision': round(precision_score(y, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y, y_pred, zero_division=0), 4),
        'auc':       round(roc_auc_score(y, y_pred), 4) if has_both else None,
        'n_rules':   len(clf.ruleset_)
    }

# ─────────────────────────────────────────────
# EXPERIMENT 1: Standard RIPPER baseline
# (same config as script 05 best model)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE: Standard RIPPER (decay=0, uniform weights)")
print("="*60)

ripper_base = RIPPER(k=2, max_rules=30, max_rule_conds=7, random_state=42)
ripper_base.fit(X_train, y_train, pos_class=1)

val_base  = evaluate(ripper_base, X_val,  y_val,  'RIPPER (standard)')
test_base = evaluate(ripper_base, X_test, y_test, 'RIPPER (standard)')

print(f"Rules: {val_base['n_rules']}")
print(f"Validation — Precision: {val_base['precision']}  Recall: {val_base['recall']}  F1: {val_base['f1']}  AUC: {val_base['auc']}")
print(f"Test       — Precision: {test_base['precision']}  Recall: {test_base['recall']}  F1: {test_base['f1']}  AUC: {test_base['auc']}")

# ─────────────────────────────────────────────
# EXPERIMENT 2: RIPPER-TW across decay values
#
# decay=0.0 should match standard RIPPER exactly
# (confirms our patch doesn't break anything)
# Then test meaningful decay values
# ─────────────────────────────────────────────
decay_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

all_results = []
all_results.append({**val_base,  'set': 'Validation'})
all_results.append({**test_base, 'set': 'Test'})

print("\n" + "="*60)
print("RIPPER-TW: Testing decay values")
print("="*60)

for decay in decay_values:
    print(f"\n--- decay={decay} ---")

    clf = RIPPER_TW(
        decay=decay,
        k=2,
        max_rules=30,
        max_rule_conds=7,
        random_state=42
    )

    clf.fit(
        X_train,
        y_train,
        time_steps=time_steps,
        pos_class=1
    )

    val_res  = evaluate(clf, X_val,  y_val,  f'RIPPER-TW (decay={decay})')
    test_res = evaluate(clf, X_test, y_test, f'RIPPER-TW (decay={decay})')

    print(f"  Rules: {val_res['n_rules']}")
    print(f"  Validation — Precision: {val_res['precision']}  Recall: {val_res['recall']}  F1: {val_res['f1']}  AUC: {val_res['auc']}")
    print(f"  Test       — Precision: {test_res['precision']}  Recall: {test_res['recall']}  F1: {test_res['f1']}  AUC: {test_res['auc']}")

    all_results.append({**val_res,  'set': 'Validation', 'decay': decay})
    all_results.append({**test_res, 'set': 'Test',       'decay': decay})

# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
results_df = pd.DataFrame(all_results)

print("\n" + "="*60)
print("FULL COMPARISON — VALIDATION SET")
print("="*60)
val_df = results_df[results_df['set'] == 'Validation']
print(val_df[['model', 'n_rules', 'precision', 'recall', 'f1', 'auc']].to_string(index=False))

print("\n" + "="*60)
print("FULL COMPARISON — TEST SET")
print("="*60)
test_df = results_df[results_df['set'] == 'Test']
print(test_df[['model', 'n_rules', 'precision', 'recall', 'f1', 'auc']].to_string(index=False))

results_df.to_csv(os.path.join(RESULTS_DIR, 'ripper_tw_results.csv'), index=False)
print(f"\nSaved: outputs/results/ripper_tw_results.csv")

# ─────────────────────────────────────────────
# Plot: F1 by decay value (val and test)
# ─────────────────────────────────────────────
tw_val  = results_df[(results_df['set']=='Validation') & (results_df['model'].str.contains('TW'))]
tw_test = results_df[(results_df['set']=='Test')       & (results_df['model'].str.contains('TW'))]

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(decay_values, tw_val['f1'].values,
        marker='o', color='steelblue', linewidth=2, label='Validation F1')
ax.plot(decay_values, tw_test['f1'].values,
        marker='s', color='salmon',   linewidth=2, label='Test F1')

# Baseline reference lines
ax.axhline(y=val_base['f1'],  color='steelblue', linestyle='--',
           alpha=0.5, label=f"Standard RIPPER val F1={val_base['f1']}")
ax.axhline(y=test_base['f1'], color='salmon',    linestyle='--',
           alpha=0.5, label=f"Standard RIPPER test F1={test_base['f1']}")

ax.set_xlabel('Decay parameter (λ)', fontsize=12)
ax.set_ylabel('F1 Score — illicit class', fontsize=12)
ax.set_title('RIPPER-TW: Effect of Temporal Decay on F1\n(Internal time-weighted FOIL gain modification)',
             fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'ripper_tw_decay_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ripper_tw_decay_comparison.png")

print("\n=== STEP 12 COMPLETE: RIPPER-TW Experiment Done ===")