import pandas as pd
import numpy as np
import os
import os
DATA_DIR = 'data/'

FEATURES_PATH = os.path.join(DATA_DIR, 'elliptic_txs_features.csv')
CLASSES_PATH = os.path.join(DATA_DIR, 'elliptic_txs_classes.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'labeled_data.csv')

features=pd.read_csv(FEATURES_PATH, header=None)

cols=['id','time_step']+[f'f{i}' for i in range(1, 166)]
features.columns = cols


classes = pd.read_csv(CLASSES_PATH)
classes = classes.rename(columns={'txId':'id'})
classes['id'] = classes['id'].astype(int)

# Merge features and classes on id
df = features.merge(classes, on='id')
# Remove unknown labels
df = df[df['class'] != 'unknown']

df['class'] = df['class'].astype(int)
df['class'] = df['class'].replace({2: 0})
df = df.reset_index(drop=True)

# Save to file
df.to_csv(OUTPUT_PATH, index=False)

# Print summary
print("=== STEP 1 COMPLETE: Load and Merge ===")
print(f"Shape: {df.shape}")
print(f"\nClass distribution:")
print(df['class'].value_counts())
print(f"\nIllicit: {(df['class']==1).sum()}")
print(f"Licit: {(df['class']==0).sum()}")
print(f"Imbalance ratio: {round((df['class']==0).sum() / (df['class']==1).sum(), 2)}:1")
print(f"\nTime steps: {df['time_step'].nunique()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nSaved to: {OUTPUT_PATH}")