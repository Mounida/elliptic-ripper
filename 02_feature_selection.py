import pandas as pd
import numpy as np
import os

DATA_DIR = 'data/'
INPUT_PATH = os.path.join(DATA_DIR, 'labeled_data.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'selected_features.csv')

df=pd.read_csv(INPUT_PATH)
feature_cols = [f'f{i}' for i in range(1, 166)]
print(f"Starting features: {len(feature_cols)}")

# Filter 1: Remove low variance features
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
selector.fit(df[feature_cols])

# Get the names of features that passed the variance threshold filter (variance >= 0.01)   
high_variance_cols = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]

print(f"After variance filter: {len(high_variance_cols)} features")
print(f"Removed: {len(feature_cols) - len(high_variance_cols)} features")

# Filter 2: Remove highly correlated features
corr_matrix = df[high_variance_cols].corr().abs()
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
# Find columns where correlation > 0.95
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

# Keep the rest
low_corr_cols = [col for col in high_variance_cols if col not in to_drop]

print(f"After correlation filter: {len(low_corr_cols)} features")
print(f"Removed: {len(to_drop)} features")

final_cols = ['id', 'time_step'] + low_corr_cols + ['class']
df_selected = df[final_cols]
df_selected.to_csv(OUTPUT_PATH, index=False)

print(f"\n=== STEP 2 COMPLETE: Feature Selection ===")
print(f"Original features: {len(feature_cols)}")
print(f"After variance filter: {len(high_variance_cols)}")
print(f"After correlation filter: {len(low_corr_cols)}")
print(f"Total features removed: {len(feature_cols) - len(low_corr_cols)}")
print(f"\nFinal dataset shape: {df_selected.shape}")
print(f"\nSaved to: {OUTPUT_PATH}")

