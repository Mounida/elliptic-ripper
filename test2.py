import pandas as pd
import numpy as np

# Load data
features = pd.read_csv('elliptic_txs_features.csv', header=None)
classes = pd.read_csv('elliptic_txs_classes.csv', header=None,
                      names=['id', 'class'])

# Remove any non-numeric IDs (like 'txId')
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

print("=== WORKING DATASET (labeled only) ===")
print("Shape:", labeled.shape)
print("\nClass distribution:")
print(labeled['class'].value_counts())

# Imbalance ratio (dynamic)
counts = labeled['class'].value_counts()
ratio = round(counts[2] / counts[1], 2)
print("\nImbalance ratio:", ratio, ": 1")

print("\n=== TIME STEPS ===")
print("Number of time steps:", labeled['time_step'].nunique())
print("Time step distribution (first 5):")
print(labeled['time_step'].value_counts().sort_index().head())

print("\n=== MISSING VALUES ===")
missing = labeled.isnull().sum().sum()
print("Total missing values:", missing)

print("\n=== FEATURE RANGES ===")
print("Feature min:", round(labeled.iloc[:, 2:].min().min(), 4))
print("Feature max:", round(labeled.iloc[:, 2:].max().max(), 4))
print("Already normalized:", "Yes" if labeled.iloc[:, 2:].max().max() < 100 else "Check manually")