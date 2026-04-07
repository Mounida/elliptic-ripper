import pandas as pd
import os

DATA_DIR = 'data/'
features_path= os.path.join(DATA_DIR, 'elliptic_txs_features.csv')
classes_path= os.path.join(DATA_DIR, 'elliptic_txs_classes.csv')
edges_path= os.path.join(DATA_DIR, 'elliptic_txs_edgelist.csv')

# Load the three files
features = pd.read_csv(features_path, header=None)
classes = pd.read_csv(classes_path)
edges = pd.read_csv(edges_path, header=None,
                    names=['source', 'target'])

# Fix column names
features.rename(columns={0: 'id'}, inplace=True)
classes.columns = ['id', 'class']

# Remove possible duplicated header row
classes = classes[classes['id'] != 'id']

# Ensure correct data types
features['id'] = features['id'].astype(int)
classes['id'] = classes['id'].astype(int)

# Basic info
print("=== FEATURES FILE ===")
print("Shape:", features.shape)
print("\nFirst 3 rows:")
print(features.head(3))

print("\n=== CLASSES FILE ===")
print("Shape:", classes.shape)
print("\nClass distribution:")
print(classes['class'].value_counts())

print("\n=== EDGES FILE ===")
print("Shape:", edges.shape)

# Merge features and classes
df = features.merge(classes, on='id')

print("\n=== MERGED DATASET ===")
print("Shape:", df.shape)
print("\nClass distribution in merged data:")
print(df['class'].value_counts())