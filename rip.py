import pandas as pd
import wittgenstein as lw
from sklearn.metrics import classification_report, accuracy_score
print("SCRIPT START")
# =========================
# 1. LOAD DATA
# =========================
DATA_DIR = "data/"

train = pd.read_csv(DATA_DIR + "train.csv")
val = pd.read_csv(DATA_DIR + "val.csv")
test = pd.read_csv(DATA_DIR + "test.csv")

# =========================
# 2. SPLIT FEATURES / LABEL
# =========================
X_train = train.drop("class", axis=1)
y_train = train["class"]

X_val = val.drop("class", axis=1)
y_val = val["class"]

X_test = test.drop("class", axis=1)
y_test = test["class"]

print("=== DATA LOADED ===")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# =========================
# 3. TRAIN RIPPER MODEL
# =========================
model = lw.RIPPER(
    random_state=42,
    max_rules=100,     # limite de règles
    prune_size=0.3     # pruning (réduction surapprentissage)
)

model.fit(X_train, y_train)

# =========================
# 4. PREDICTIONS
# =========================
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

# =========================
# 5. EVALUATION
# =========================
print("\n=== VALIDATION RESULTS ===")
print(classification_report(y_val, val_pred))
print("Accuracy:", accuracy_score(y_val, val_pred))

print("\n=== TEST RESULTS ===")
print(classification_report(y_test, test_pred))
print("Accuracy:", accuracy_score(y_test, test_pred))

# =========================
# 6. RULES (IMPORTANT 🔥)
# =========================
print("\n=== RIPPER RULES ===")
print(model.ruleset_)