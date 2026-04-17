import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Paths
DATA_DIR = 'data/'
VAL_PATH = os.path.join(DATA_DIR, 'val.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'ripper_model.pkl')

# Load model
with open(MODEL_PATH, 'rb') as f:
    clf = pickle.load(f)

print("Model loaded successfully.")
print(f"Number of rules: {len(clf.ruleset_)}")

# Load validation and test sets
val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

# Feature columns
feature_cols = [col for col in val.columns if col.startswith('f')]

# Separate features and labels
X_val = val[feature_cols]
y_val = val['class']

X_test = test[feature_cols]
y_test = test['class']

# Predict
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)
# Convert boolean predictions to integers
y_val_pred = pd.Series(y_val_pred).astype(int)
y_test_pred = pd.Series(y_test_pred).astype(int)
# Evaluation function
def evaluate(y_true, y_pred, set_name):
    print(f"\n=== {set_name} ===")
    print(f"Total transactions: {len(y_true)}")
    print(f"Actual illicit: {(y_true == 1).sum()}")
    print(f"Predicted illicit: {(y_pred == 1).sum()}")
    print(f"\nPrecision (illicit): {round(precision_score(y_true, y_pred, zero_division=0), 4)}")
    print(f"Recall (illicit):    {round(recall_score(y_true, y_pred, zero_division=0), 4)}")
    print(f"F1 Score (illicit):  {round(f1_score(y_true, y_pred, zero_division=0), 4)}")
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  True Licit (TN):     {cm[0][0]}")
    print(f"  False Illicit (FP):  {cm[0][1]}")
    print(f"  Missed Fraud (FN):   {cm[1][0]}")
    print(f"  Caught Fraud (TP):   {cm[1][1]}")

# Run evaluation
evaluate(y_val, y_val_pred, "VALIDATION SET")
evaluate(y_test, y_test_pred, "TEST SET")
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
plt.title("RIPPER Confusion Matrix")
plt.show()
print("\n=== STEP 6 COMPLETE: Evaluation Done ===")