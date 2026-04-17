import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from wittgenstein import RIPPER

DATA_DIR    = 'data/'
RESULTS_DIR = 'outputs/results/'
FIGURES_DIR = 'outputs/figures/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# We need the full timeline in one place.
# We reconstruct it by merging train + val + test
# in chronological order.
# ─────────────────────────────────────────────

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
val   = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

full_data    = pd.concat([train, val, test], ignore_index=True)
feature_cols = [col for col in train.columns if col.startswith('f')]
all_steps    = sorted(full_data['time_step'].unique())

print(f"Full dataset: {len(full_data)} transactions")
print(f"Time steps: {min(all_steps)} to {max(all_steps)}")
print(f"Features: {len(feature_cols)}")

# ─────────────────────────────────────────────
# SHARED HELPER: balance a training window
#
# This recalculates the imbalance ratio LOCALLY
# for whatever window of data you pass in.
# This is important — the global class weights
# from step 4 were calculated on the full training
# set. A smaller window has a different ratio,
# so we must recalculate each time.
# ─────────────────────────────────────────────
def balance_window(window_df, feature_cols, random_state=42):
    illicit = window_df[window_df['class'] == 1]
    licit   = window_df[window_df['class'] == 0]

    n_illicit = len(illicit)
    n_licit   = len(licit)

    if n_illicit == 0:
        return None, None  # can't train with zero positives

    # Repeat illicit samples to approximately match licit count
    # Cap at 8 repetitions to avoid extreme memory use on small windows
    repeat = max(1, min(8, round(n_licit / n_illicit)))

    illicit_rep = pd.concat([illicit] * repeat, ignore_index=True)
    balanced    = pd.concat([illicit_rep, licit], ignore_index=True)
    balanced    = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    X = balanced[feature_cols]
    y = balanced['class']
    return X, y
# ─────────────────────────────────────────────
# SHARED HELPER: train RIPPER and evaluate
# on a single test step
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, y_train, X_test, y_test, step):
    """
    Trains a fresh RIPPER on (X_train, y_train),
    predicts on (X_test, y_test),
    returns a dict of metrics.
    Returns None if training is not possible.
    """
    if X_train is None or (y_train == 1).sum() < 10:
        return None  # not enough positives to learn rules

    clf = RIPPER(k=2, max_rules=30, max_rule_conds=7, random_state=42)
    try:
        clf.fit(X_train, y_train, pos_class=1)
    except Exception as e:
        print(f"  Step {step}: RIPPER failed — {e}")
        return None

    y_pred = pd.Series(clf.predict(X_test)).astype(int)

    # AUC requires both classes to be present in y_test
    has_both_classes = len(y_test.unique()) == 2

    return {
        'predict_step': step,
        'n_train':      len(X_train),
        'n_illicit_train': int((y_train == 1).sum()),
        'n_test':       len(X_test),
        'n_illicit_test': int((y_test == 1).sum()),
        'n_rules':      len(clf.ruleset_),
        'f1':           round(f1_score(y_test, y_pred, zero_division=0), 4),
        'precision':    round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':       round(recall_score(y_test, y_pred, zero_division=0), 4),
        'auc':          round(roc_auc_score(y_test, y_pred), 4) if has_both_classes else None
    }
# ─────────────────────────────────────────────
# STRATEGY 1: SLIDING WINDOW
#
# For each step T we want to predict:
#   - Training data = steps [T - W, T - 1]
#   - Test data     = step T only
#   - The window slides forward one step at a time
#
# We start predicting from step WINDOW_SIZE + 1
# because we need at least W steps of history
# to form the first window.
# ─────────────────────────────────────────────
WINDOW_SIZE = 15

print("\n" + "="*55)
print(f"STRATEGY 1: SLIDING WINDOW (W={WINDOW_SIZE})")
print("="*55)

sliding_results = []

# We predict on every step from WINDOW_SIZE+1 onward
# The first W steps are consumed as the initial window
for i in range(WINDOW_SIZE, len(all_steps)):
    predict_step = all_steps[i]
    train_steps  = all_steps[i - WINDOW_SIZE : i]  # exactly W steps before

    window_data = full_data[full_data['time_step'].isin(train_steps)]
    step_data   = full_data[full_data['time_step'] == predict_step]

    # Skip if no illicit in test step — metrics are undefined
    if (step_data['class'] == 1).sum() == 0:
        print(f"  Step {predict_step}: skipped (no illicit in test step)")
        continue

    X_w, y_w = balance_window(window_data, feature_cols)
    X_s = step_data[feature_cols]
    y_s = step_data['class']

    result = train_and_evaluate(X_w, y_w, X_s, y_s, predict_step)

    if result:
        result['train_range'] = f"{min(train_steps)}-{max(train_steps)}"
        sliding_results.append(result)
        print(f"  Step {predict_step:2d} | trained on {result['train_range']} | "
              f"illicit_train={result['n_illicit_train']:3d} | "
              f"illicit_test={result['n_illicit_test']:3d} | "
              f"F1={result['f1']:.4f} | AUC={str(result['auc'])}")

sliding_df = pd.DataFrame(sliding_results)
sliding_df.to_csv(os.path.join(RESULTS_DIR, 'sliding_window_results.csv'), index=False)

avg_f1_sliding = round(sliding_df['f1'].mean(), 4)
print(f"\nSliding window — mean F1 across all steps: {avg_f1_sliding}")
# ─────────────────────────────────────────────
# STRATEGY 2: INCREMENTAL LEARNING
#
# For each step T we want to predict:
#   - Training data = ALL steps from 1 up to T-1
#   - Test data     = step T only
#   - Training set grows by one step each iteration
#
# We start predicting from step 2 (need at least
# step 1 as training history).
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("STRATEGY 2: INCREMENTAL (growing window)")
print("="*55)

incremental_results = []

for i in range(1, len(all_steps)):
    predict_step = all_steps[i]
    train_steps  = all_steps[:i]  # all steps before predict_step

    cumul_data = full_data[full_data['time_step'].isin(train_steps)]
    step_data  = full_data[full_data['time_step'] == predict_step]

    if (step_data['class'] == 1).sum() == 0:
        print(f"  Step {predict_step}: skipped (no illicit in test step)")
        continue

    X_c, y_c = balance_window(cumul_data, feature_cols)
    X_s = step_data[feature_cols]
    y_s = step_data['class']

    result = train_and_evaluate(X_c, y_c, X_s, y_s, predict_step)

    if result:
        result['train_range'] = f"1-{max(train_steps)}"
        incremental_results.append(result)
        print(f"  Step {predict_step:2d} | trained on {result['train_range']:>6} | "
              f"illicit_train={result['n_illicit_train']:4d} | "
              f"illicit_test={result['n_illicit_test']:3d} | "
              f"F1={result['f1']:.4f} | AUC={str(result['auc'])}")

incremental_df = pd.DataFrame(incremental_results)
incremental_df.to_csv(os.path.join(RESULTS_DIR, 'incremental_results.csv'), index=False)

avg_f1_incremental = round(incremental_df['f1'].mean(), 4)
print(f"\nIncremental — mean F1 across all steps: {avg_f1_incremental}")
# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("STRATEGY COMPARISON SUMMARY")
print("="*55)

# Filter to only the test period steps (43-49)
# so we compare fairly against the static model's test results
test_steps = list(range(43, 50))

sliding_test   = sliding_df[sliding_df['predict_step'].isin(test_steps)]
incremental_test = incremental_df[incremental_df['predict_step'].isin(test_steps)]

print(f"\nOn test period steps (43-49):")
print(f"  Static RIPPER     — F1: 0.0185  AUC: 0.4867  (from 06_evaluate.py)")
print(f"  Sliding window    — F1: {round(sliding_test['f1'].mean(), 4):.4f}  "
      f"AUC: {round(sliding_test['auc'].dropna().mean(), 4):.4f}")
print(f"  Incremental       — F1: {round(incremental_test['f1'].mean(), 4):.4f}  "
      f"AUC: {round(incremental_test['auc'].dropna().mean(), 4):.4f}")
# ─────────────────────────────────────────────
# PLOT: All three strategies on the same graph
# Steps 35-49 (val + test period)
# This is the central figure of Chapter 3
# ─────────────────────────────────────────────

# Load static per-step results if available from 08
static_path = os.path.join(RESULTS_DIR, 'baseline_comparison.csv')

fig, ax = plt.subplots(figsize=(13, 6))

# Sliding window
ax.plot(sliding_df['predict_step'], sliding_df['f1'],
        marker='o', color='green', linewidth=2, label=f'Sliding Window (W={WINDOW_SIZE})')

# Incremental
ax.plot(incremental_df['predict_step'], incremental_df['f1'],
        marker='s', color='darkorange', linewidth=2, label='Incremental (growing window)')

# Static RIPPER baseline — flat line at 0.0185 across test steps
# We draw it only for the val+test period for visual clarity
static_steps = sorted(pd.concat([val, test])['time_step'].unique())
ax.hlines(y=0.7399, xmin=min(static_steps), xmax=42,
          colors='steelblue', linestyles='--', linewidth=1.5)
ax.hlines(y=0.0185, xmin=43, xmax=max(static_steps),
          colors='steelblue', linestyles='--', linewidth=1.5,
          label='Static RIPPER (val=0.74 / test=0.019)')

ax.axvline(x=35, color='orange', linestyle=':', linewidth=1.5, label='Validation begins (35)')
ax.axvline(x=43, color='red',    linestyle=':', linewidth=1.5, label='Test begins (43)')

ax.set_xlabel('Time Step (predicted)', fontsize=12)
ax.set_ylabel('F1 Score — illicit class', fontsize=12)
ax.set_title('Adaptive RIPPER Strategies vs Static Baseline\nF1 Per Predicted Time Step',
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'adaptive_strategies_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: adaptive_strategies_comparison.png")

print("\n=== STEP 9 COMPLETE: Adaptive RIPPER Experiment Done ===")