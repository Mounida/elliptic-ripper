import pandas as pd
import numpy as np
import os
import json

DATA_DIR = 'data/'
INPUT_PATH = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'class_weights.json')

train=pd.read_csv(INPUT_PATH)

total = len(train)
n_classes = 2
n_illicit = (train['class'] == 1).sum()
n_licit = (train['class'] == 0).sum()

# Calculate class weights
weight_illicit = round(total / (n_classes * n_illicit), 4)
weight_licit = round(total / (n_classes * n_licit), 4)

class_weights = {
    1: weight_illicit,
    0: weight_licit
}
with open(OUTPUT_PATH, 'w') as f:
    json.dump(class_weights, f)

# Print summary
print("=== STEP 4 COMPLETE: Class Weights ===")
print(f"\nTraining set:")
print(f"  Total transactions: {total}")
print(f"  Illicit (1): {n_illicit}")
print(f"  Licit (0): {n_licit}")
print(f"  Imbalance ratio: {round(n_licit / n_illicit, 2)}:1")
print(f"\nClass weights:")
print(f"  Illicit (1): {weight_illicit}")
print(f"  Licit (0): {weight_licit}")
print(f"\nSaved to: {OUTPUT_PATH}")