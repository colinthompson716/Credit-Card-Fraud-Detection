import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# get the path to the data file relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_file = os.path.join(project_root, 'data', 'raw', 'credit_card_frauds.csv')

df = pd.read_csv(csv_file)

# setting the styling for better looking graphs/plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("LOADING THE DATASET")
print("="*80)

# loading the csv file 
try: 
    df = pd.read_csv(csv_file)
    print("Dataset loaded sucsessfully")
except FileNotFoundError:
    print("ERROR: Could not find the dataset")
    print("Make sure the dataset is in the /data/raw folder")
    print("Expected path: data/raw/creditcard_fraud.csv")

# how many rows (transactions) and columns (features)
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Translation: {df.shape[0]:,} transactions with {df.shape[1]} features")

# columns (features): what information do they provide?
print("\nColumn names and data types:")
print(df.dtypes)

# first couple of rows (transactions): what does the data look like?
print("\nFirst 5 rows (transactions):")
print(df.head())

# getting basic statistics of the data like mean, std, min, and max
print("\nBasic Statistics:")
print(df.describe())

print("\nEXPLANATION OF STATISTICS:")
print("- count: Number of non-missing values")
print("- mean: Average value")
print("- std: How spread out the values are")
print("- min/max: Smallest and largest values")
print("- 25%, 50%, 75%: Quartiles (dividing the data into 4 parts)")

missing = df.isnull().sum()
print(f"\nMissing values per column:")
print(missing)

# count how many fraud vs lefitimate transactions
fraud_counts = df['is_fraud'].value_counts()
print(f"\nFraud distribution:")
print(fraud_counts)

fraud_percentage = (df['is_fraud'].sum() / len(df)) * 100
print(f"\nFraud percentage: {fraud_percentage:.2f}%")
print(f"Legitimate percentage: {100 - fraud_percentage:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# bar chart 
fraud_counts.plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Fraud vs Legitimate Transactions (Count)')
axes[0].set_xlabel('Fraud (0=No, 1=Yes)')
axes[0].set_ylabel('Number of Transactions')
axes[0].set_xticklabels(['Legitimate', 'Fraudulent'], rotation=0)

# pie chart
axes[1].pie([fraud_counts[0], fraud_counts[1]], 
           labels=['Legitimate', 'Fraudulent'],
           autopct='%1.2f%%',
           colors=['green', 'red'])
axes[1].set_title('Fraud Distribution (%)')
 
plt.tight_layout()
plt.savefig('../results/plots/01_fraud_distribution.png', dpi=100, bbox_inches='tight')
print("Saved plot to results/plots/01_fraud_distribution.png")
plt.close()

# understanding each feature
print(f"\nFeature breakdowns")
for col in df.columns:
    if col != 'is_fraud':
        print(f"\n{col}:")
        print(f"  Data type: {df[col].dtype}")
        if df[col].dtype == 'object':
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Examples: {df[col].unique()[:5]}")
        else:
            print(f"  Range: {df[col].min()} to {df[col].max()}")
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"  Mean: {df[col].mean():.2f}")

print("\nAre fraudulent transactions different from legitimate ones?")
print("\n(This helps us understand what patterns the model should learn)\n")

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'is_fraud' in numerical_cols:
    numerical_cols.remove('is_fraud')

for col in numerical_cols[:5]:
    fraud_mean = df[df['is_fraud'] == 1][col].mean()
    legit_mean = df[df['is_fraud'] == 0][col].mean()

    print(f"{col}:")
    print(f"  Legitimate avg: {legit_mean:.2f}")
    print(f"  Fraudulent avg: {fraud_mean:.2f}")
    print(f"  Difference: {abs(fraud_mean - legit_mean):.2f}")
    print()

    # visualizing comparison for a key feature (like transaction amount)
if 'amt' in df.columns:
    amount_col = 'amt'
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # distribution of amounts for legitimate transactions
    df[df['is_fraud'] == 0][amount_col].hist(bins=50, ax=axes[0], color='green', alpha=0.7)
    axes[0].set_title('Transaction Amount Distribution (Legitimate)')
    axes[0].set_xlabel('Amount')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(0, 1500)
    
    # distribution of amounts for fraudulent transactions
    df[df['is_fraud'] == 1][amount_col].hist(bins=50, ax=axes[1], color='red', alpha=0.7)
    axes[1].set_title('Transaction Amount Distribution (Fraudulent)')
    axes[1].set_xlabel('Amount')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(0, 1500)
    
    plt.tight_layout()
    plt.savefig('../results/plots/02_amount_comparison.png', dpi=100, bbox_inches='tight')
    print("Saved amount comparison plot to results/plots/02_amount_comparison.png")
    plt.close()

print(f"""
Dataset Summary:
- Total transactions: {len(df):,}
- Number of features: {df.shape[1]}
- Fraudulent transactions: {df['is_fraud'].sum():,} ({fraud_percentage:.2f}%)
- Legitimate transactions: {len(df) - df['is_fraud'].sum():,} ({100-fraud_percentage:.2f}%)
- Missing values: {missing.sum()}
 
Key Challenges:
1. CLASS IMBALANCE: Fraud is rare (~{fraud_percentage:.2f}%)
     Solution: Use resampling or cost-weighted learning
   
2. IMBALANCED METRICS: Accuracy is misleading
     Solution: Use Precision, Recall, and ROC-AUC instead
 
3. FEATURE ENGINEERING: Need to create meaningful features
     Solution: Create features like amount deviation from average
""")