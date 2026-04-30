import pandas as pd
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

# Paths
DATA_DIR    = 'data/'
RESULTS_DIR = 'outputs/results/'
FIGURES_DIR = 'outputs/figures/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
val   = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

with open(os.path.join(DATA_DIR, 'class_weights.json'), 'r') as f:
    class_weights = json.load(f)

with open(os.path.join(DATA_DIR, 'ripper_model.pkl'), 'rb') as f:
    ripper_clf = pickle.load(f)

feature_cols = [col for col in train.columns if col.startswith('f')]

# Build balanced training set
weight_illicit = class_weights['1']
weight_licit   = class_weights['0']
repeat_times   = round(weight_illicit / weight_licit)

illicit_train  = train[train['class'] == 1]
licit_train    = train[train['class'] == 0]
illicit_rep    = pd.concat([illicit_train] * repeat_times, ignore_index=True)
train_balanced = pd.concat([illicit_rep, licit_train], ignore_index=True)
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_train = train_balanced[feature_cols]
y_train = train_balanced['class']
X_val   = val[feature_cols];  y_val  = val['class']
X_test  = test[feature_cols]; y_test = test['class']

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"Balanced training set: {len(X_train)}")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_lr = scaler.fit_transform(X_train)
X_val_lr   = scaler.transform(X_val)
X_test_lr  = scaler.transform(X_test)
# Define models
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

# Train and save models
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == 'Logistic Regression':
        model.fit(X_train_lr, y_train)
    else:
        model.fit(X_train, y_train)    
    trained_models[name] = model
    filename = name.lower().replace(' ', '_') + '_model.pkl'
    with open(os.path.join(DATA_DIR, filename), 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {filename}")

# Evaluation function
def evaluate(model, X, y, model_name, set_name):
    y_pred = pd.Series(model.predict(X)).astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'model':     model_name,
        'set':       set_name,
        'precision': round(precision_score(y, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y, y_pred, zero_division=0), 4),
        'auc':       round(roc_auc_score(y, y_pred), 4),
        'TP': int(tp), 'FP': int(fp),
        'TN': int(tn), 'FN': int(fn)
    }
# Evaluate all models including RIPPER
all_models = {'RIPPER': ripper_clf, **trained_models}
all_results = []

for model_name, model in all_models.items():
    for X, y, set_name in [(X_val, y_val, 'Validation'),
                            (X_test, y_test, 'Test')]:
        all_results.append(evaluate(model, X, y, model_name, set_name))

results_df = pd.DataFrame(all_results)

# Print results
for set_name in ['Validation', 'Test']:
    print(f"\n{'='*65}")
    print(f"RESULTS — {set_name.upper()} SET")
    print(f"{'='*65}")
    subset = results_df[results_df['set'] == set_name]
    print(subset[['model', 'precision', 'recall', 'f1', 'auc',
                  'TP', 'FP', 'FN']].to_string(index=False))

results_df.to_csv(os.path.join(RESULTS_DIR, 'baseline_comparison.csv'), index=False)
print(f"\nSaved: outputs/results/baseline_comparison.csv")

# Plot: Validation vs Test F1
model_names = list(all_models.keys())
val_f1s  = [results_df[(results_df['model']==m) & (results_df['set']=='Validation')]['f1'].values[0] for m in model_names]
test_f1s = [results_df[(results_df['model']==m) & (results_df['set']=='Test')]['f1'].values[0] for m in model_names]

x     = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, val_f1s,  width, label='Validation F1', color='steelblue')
bars2 = ax.bar(x + width/2, test_f1s, width, label='Test F1',       color='salmon')

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel('F1 Score (classe illicite)')
ax.set_title('Comparaison des modèles — Validation vs Test F1')
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'baseline_f1_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: baseline_f1_comparison.png")

print("\n=== STEP 8 COMPLETE: Baseline Comparison Done ===")