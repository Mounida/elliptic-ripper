import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import re

DATA_DIR    = 'data/'
RESULTS_DIR = 'outputs/results/'
FIGURES_DIR = 'outputs/figures/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load the trained RIPPER model
with open(os.path.join(DATA_DIR, 'ripper_model.pkl'), 'rb') as f:
    clf = pickle.load(f)

print(f"Model loaded.")
print(f"Number of rules: {len(clf.ruleset_)}")
print(f"\n--- RAW RULES ---")
print(clf.ruleset_)
# Convert the ruleset to a plain string so we can search through it
rules_text = str(clf.ruleset_)

# re.findall searches the text for every occurrence of the pattern
# The pattern r'f\d+' means: the letter f followed by one or more digits
# This matches f41, f2, f161, f100 etc.
all_features_mentioned = re.findall(r'f\d+', rules_text)

print(f"\nTotal feature mentions across all rules: {len(all_features_mentioned)}")
print(f"Unique features used: {len(set(all_features_mentioned))}")

# Count how many times each feature appears
feature_counts = {}
for feature in all_features_mentioned:
    if feature not in feature_counts:
        feature_counts[feature] = 0
    feature_counts[feature] += 1

# Sort by count, highest first
feature_counts_sorted = dict(
    sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
)

print(f"\n--- FEATURE FREQUENCY IN RULES ---")
for feature, count in feature_counts_sorted.items():
    bar = '█' * count
    print(f"  {feature:<6} appears {count:2d} times  {bar}")

# ─────────────────────────────────────────────
# PLOT 1: Top 15 most frequent features
# ─────────────────────────────────────────────

# Take only the top 15 for a clean readable chart
top_n = 15
top_features = list(feature_counts_sorted.keys())[:top_n]
top_counts   = list(feature_counts_sorted.values())[:top_n]

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.barh(top_features[::-1], top_counts[::-1], color='steelblue')
# [::-1] reverses the list so the highest bar appears at the top

# Add count labels at the end of each bar
for bar, count in zip(bars, top_counts[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=10)

ax.set_xlabel('Number of times feature appears across all 25 rules', fontsize=11)
ax.set_title('Most Frequently Used Features in RIPPER Rules\n(Top 15 out of 43 features used)',
             fontsize=12)
ax.axvline(x=25, color='red', linestyle='--', alpha=0.4, label='Total rules = 25')
ax.legend()
ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'top_features_frequency.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: top_features_frequency.png")
# ─────────────────────────────────────────────
# RULE COMPLEXITY ANALYSIS
# How many conditions does each rule have?
# ─────────────────────────────────────────────

# Split the full ruleset text into individual rules
# Each rule is wrapped in [ ] so we split on ] V [
rules_raw = rules_text.strip('[]').split('] V [')

rule_lengths = []
for i, rule in enumerate(rules_raw):
    # Count conditions by counting ^ (AND operator)
    # A rule with 3 conditions has 2 ^ symbols
    # So conditions = number of ^ + 1
    n_conditions = rule.count('^') + 1
    rule_lengths.append(n_conditions)
    print(f"  Rule {i+1:2d}: {n_conditions} conditions")

print(f"\nAverage conditions per rule: {round(sum(rule_lengths)/len(rule_lengths), 2)}")
print(f"Simplest rule: {min(rule_lengths)} condition(s)")
print(f"Most complex rule: {max(rule_lengths)} conditions")
# ─────────────────────────────────────────────
# PLOT 2: Rule complexity distribution
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(range(1, len(rule_lengths)+1), rule_lengths, color='salmon', edgecolor='white')
ax.axhline(y=sum(rule_lengths)/len(rule_lengths), color='steelblue',
           linestyle='--', linewidth=1.5, label=f'Average = {round(sum(rule_lengths)/len(rule_lengths),1)}')

ax.set_xlabel('Rule number', fontsize=11)
ax.set_ylabel('Number of conditions', fontsize=11)
ax.set_title('Number of Conditions per RIPPER Rule', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'rule_complexity.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: rule_complexity.png")

# Save feature frequency to CSV for thesis reference
freq_df = pd.DataFrame({
    'feature': list(feature_counts_sorted.keys()),
    'count':   list(feature_counts_sorted.values())
})
freq_df.to_csv(os.path.join(RESULTS_DIR, 'feature_frequency.csv'), index=False)
print("Saved: feature_frequency.csv")

print("\n=== STEP 10 COMPLETE: Rule Analysis Done ===")