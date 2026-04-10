import pandas as pd
import os

DATA_DIR = 'data/'
INPUT_PATH = os.path.join(DATA_DIR, 'selected_features.csv')

df = pd.read_csv(INPUT_PATH)

#split into train, val, test based on time_step
# close to 70% train, 15% val, 15% test
train = df[df['time_step'] <= 34].reset_index(drop=True)
val = df[(df['time_step'] >= 35) & (df['time_step'] <= 42)].reset_index(drop=True)
test = df[df['time_step'] >= 43].reset_index(drop=True)

#save to csv files
train.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
val.to_csv(os.path.join(DATA_DIR, 'val.csv'), index=False)
test.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
print("\nCheck no overlap:")
print("Train/Test overlap:", set(train['id']).intersection(set(test['id'])))
print("=== STEP 3 COMPLETE: Temporal Split ===")
print(f"\nTrain (steps 1-34):      {len(train)} transactions")
print(f"  Illicit: {(train['class']==1).sum()}")
print(f"  Licit:   {(train['class']==0).sum()}")
print(f"  Imbalance ratio: {round((train['class']==0).sum() / (train['class']==1).sum(), 2)}:1")

print(f"\nValidation (steps 35-42): {len(val)} transactions")
print(f"  Illicit: {(val['class']==1).sum()}")
print(f"  Licit:   {(val['class']==0).sum()}")
print(f"  Imbalance ratio: {round((val['class']==0).sum() / (val['class']==1).sum(), 2)}:1")

print(f"\nTest (steps 43-49):       {len(test)} transactions")
print(f"  Illicit: {(test['class']==1).sum()}")
print(f"  Licit:   {(test['class']==0).sum()}")
print(f"  Imbalance ratio: {round((test['class']==0).sum() / (test['class']==1).sum(), 2)}:1")

print(f"\nSaved: train.csv, val.csv, test.csv")