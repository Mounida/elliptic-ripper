import pandas as pd
import numpy as np

# Load data
features = pd.read_csv('elliptic_txs_features.csv', header=None)
classes = pd.read_csv('elliptic_txs_classes.csv', header=None,
                      names=['id', 'class'])

# 🔥 CLEAN classes (this is the key fix)
classes = classes[pd.to_numeric(classes['id'], errors='coerce').notnull()]

# Convert types
features[0] = features[0].astype(int)
classes['id'] = classes['id'].astype(int)

# Rename columns
cols = ['id', 'time_step'] + [f'f{i}' for i in range(1, 166)]
features.columns = cols

# Merge
df = features.merge(classes, on='id')

# Keep only labeled data
labeled = df[df['class'] != 'unknown'].copy()
labeled['class'] = labeled['class'].astype(int)

# Time step split preview
train = labeled[labeled['time_step'] <= 34]
val = labeled[(labeled['time_step'] >= 35) & (labeled['time_step'] <= 42)]
test = labeled[labeled['time_step'] >= 43]

print("=== TEMPORAL SPLIT PREVIEW ===")
print(f"Train (steps 1-34): {len(train)} transactions")
print(f"  Illicit: {(train['class']==1).sum()}")
print(f"  Licit: {(train['class']==2).sum()}")

print(f"\nValidation (steps 35-42): {len(val)} transactions")
print(f"  Illicit: {(val['class']==1).sum()}")
print(f"  Licit: {(val['class']==2).sum()}")

print(f"\nTest (steps 43-49): {len(test)} transactions")
print(f"  Illicit: {(test['class']==1).sum()}")
print(f"  Licit: {(test['class']==2).sum()}")

print("\n=== FEATURE VARIANCE ===")
feature_cols = [f'f{i}' for i in range(1, 166)]
variances = labeled[feature_cols].var()

low_var = (variances < 0.01).sum()
print(f"Features with variance < 0.01: {low_var}")
print(f"Features with variance >= 0.01: {(variances >= 0.01).sum()}")

print("\n=== CORRELATION CHECK ===")
corr_matrix = labeled[feature_cols].corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr = (upper > 0.95).sum().sum()
print(f"Feature pairs with correlation > 0.95: {high_corr}")