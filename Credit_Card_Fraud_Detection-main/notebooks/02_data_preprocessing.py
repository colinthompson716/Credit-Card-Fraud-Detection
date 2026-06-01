import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle
import os
from datetime import datetime

# get the path to the data file relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_file = os.path.join(project_root, 'data', 'raw', 'credit_card_frauds.csv')

# setting the styling for better looking graphs/plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("\nLOADING THE DATASET")
print("-"*80)

try:
    df = pd.read_csv(csv_file)
    print("Dataset loaded successfully!")
    print(f"  Shape: {df.shape[0]:,} transactions, {df.shape[1]} features")
except FileNotFoundError:
    print("ERROR: Could not find the dataset")
    print("  Make sure the dataset is in the /data/raw folder")
    print("  Expected path: data/raw/credit_card_frauds.csv")
    exit()

print(f"\nFirst 5 rows (transactions):")
print(df.head())

print("\n" + "="*80)
print("CHECKING FOR MISSING VALUES")
print("-"*80)

missing_values = df.isnull().sum()
print(f"\nMissing values per column:")
print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values found.")
else:
    print(f"\nWarning: {missing_values.sum()} missing values found.")
    print("Removing rows with missing values...")
    df = df.dropna()
    print(f"After removal: {df.shape[0]:,} rows remain")

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("-"*80)

print("\nExtracting features from dates and times...")

# convert transaction date/time to datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# extract time based features
df['trans_hour'] = df['trans_date_trans_time'].dt.hour
df['trans_day_of_week'] = df['trans_date_trans_time'].dt.dayofweek  # 0=Monday, 6=Sunday
df['trans_month'] = df['trans_date_trans_time'].dt.month
df['trans_is_weekend'] = (df['trans_day_of_week'] >= 5).astype(int)  # 1 if Saturday/Sunday

print("  Extracted: hour, day_of_week, month, is_weekend")

# convert DOB to age
df['dob'] = pd.to_datetime(df['dob'])
reference_date = df['trans_date_trans_time'].max()  # Use max transaction date as reference
df['age'] = (reference_date - df['dob']).dt.days / 365.25  # Convert to years

print("  Extracted: age (in years)")

# drop columns we don't need anymore
print("\nDropping unnecessary columns...")
cols_to_drop = ['trans_date_trans_time', 'dob', 'trans_num']
df = df.drop(columns=cols_to_drop)
print(f"  Dropped: {cols_to_drop}")

print(f"\nDataset shape after feature engineering: {df.shape}")
print(f"New columns: trans_hour, trans_day_of_week, trans_month, trans_is_weekend, age")

print("\n" + "="*80)
print("IDENTIFYING DATA TYPES")
print("-"*80)

print("\nColumn data types:")
print(df.dtypes)

# separate numerical and categorical features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# remove 'is_fraud' from numerical columns (it's our target, not a feature)
if 'is_fraud' in numerical_cols:
    numerical_cols.remove('is_fraud')

print("\nFeatures to use for training:")
print(f"  Numerical: {numerical_cols}")
print(f"  Categorical: {categorical_cols}")
print(f"  Target: is_fraud")

print("\n" + "="*80)
print("ENCODING CATEGORICAL FEATURES")
print("-"*80)

if categorical_cols:
    print(f"\nFound {len(categorical_cols)} categorical columns to encode...")
    print(f"  {categorical_cols}")
    
    # one-hot encoding: convert categorical data to numerical data so our models can understand it
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"Encoding complete. New shape: {df_encoded.shape}")
    print(f"  Original: {df.shape[1]} columns")
    print(f"  After encoding: {df_encoded.shape[1]} columns")
    
    df = df_encoded
else:
    print("\nNo categorical columns to encode.")

print("\n" + "="*80)
print("SEPARATING FEATURES AND TARGET")
print("-"*80)

# target is 'is_fraud' (0 = legitimate, 1 = fraud)
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"  {X.shape[1]} features, {X.shape[0]:,} transactions")

print(f"\nTarget (y) shape: {y.shape}")
print(f"  Fraud: {y.sum():,} ({(y.sum()/len(y)*100):.2f}%)")
print(f"  Legitimate: {len(y) - y.sum():,} ({((len(y)-y.sum())/len(y)*100):.2f}%)")

print("\n" + "="*80)
print("TRAIN-TEST SPLIT (80/20)")
print("-"*80)

print("\nWhat this does:")
print("  - 80% of data goes to training set (model learns from this)")
print("  - 20% of data goes to test set (model is evaluated on unseen data)")
print("\nWhy it matters:")
print("  - Test set simulates real-world performance")
print("  - If model just memorized training data, test performance will be worse")
print("  - Test results are what actually matters")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           
    random_state=42,         
    stratify=y               
)

# reset indices so they match
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"\nTraining set:")
print(f"  {X_train.shape[0]:,} transactions ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Fraud: {y_train.sum():,}")

print(f"\nTest set:")
print(f"  {X_test.shape[0]:,} transactions ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"  Fraud: {y_test.sum():,}")

print("\n" + "="*80)
print("NORMALIZING FEATURES")
print("-"*80)

print("\nWhat this does:")
print("  - Converts all features to same scale (e.g., -1 to 1)")
print("  - Example: amount $10 and amount $1000 are treated fairly")
print("\nWhy it matters:")
print("  - Without scaling, large values dominate decisions")
print("  - Decision trees don't need it, but other algorithms do")

print(f"\nBefore scaling (first 5 rows):")
print(X_train.head())

print(f"\nRange of some features:")
for col in X_train.columns[:5]:
    print(f"  {col}: {X_train[col].min():.2f} to {X_train[col].max():.2f}")

# normalize features to same scale so larger values don't dominate the model's decisions
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nAfter scaling (first 5 rows):")
print(pd.DataFrame(X_train_scaled[:5], columns=X_train.columns))

print(f"\nScaled features are now between -3 and +3")
print(f"  Mean: ~0, Standard Deviation: ~1")

# convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\n" + "="*80)
print("HANDLING CLASS IMBALANCE")
print("-"*80)

print("\nThe problem:")
print(f"  - Fraud is only {(y_train.sum()/len(y_train)*100):.2f}% of training data")
print("  - Model might learn: 'Just say everything is legitimate'")
print("  - This gets 99%+ accuracy but catches 0 frauds")

print("\nOur solution: Oversampling")
print("  - Duplicate fraud cases so fraud/legitimate are more balanced")
print("  - Makes model learn fraud patterns instead of ignoring them")

# separate fraud and legitimate in training set
fraud_train = X_train_scaled[y_train == 1]
legit_train = X_train_scaled[y_train == 0]

print(f"\nBefore resampling:")
print(f"  Fraud: {len(fraud_train)} ({len(fraud_train)/len(X_train_scaled)*100:.2f}%)")
print(f"  Legitimate: {len(legit_train)} ({len(legit_train)/len(X_train_scaled)*100:.2f}%)")

# oversample fraud cases to match legitimate count
fraud_train_resampled = resample(
    fraud_train,
    n_samples=len(legit_train),  
    random_state=42
)

# combine resampled fraud with legitimate
X_train_balanced = pd.concat([fraud_train_resampled, legit_train])
y_train_balanced = pd.concat([
    pd.Series([1] * len(fraud_train_resampled)),
    pd.Series([0] * len(legit_train))
])

# shuffle them up
shuffle_idx = np.random.RandomState(42).permutation(len(X_train_balanced))
X_train_balanced = X_train_balanced.iloc[shuffle_idx].reset_index(drop=True)
y_train_balanced = y_train_balanced.iloc[shuffle_idx].reset_index(drop=True)

print(f"\nAfter resampling:")
print(f"  Fraud: {(y_train_balanced==1).sum()} ({(y_train_balanced==1).sum()/len(y_train_balanced)*100:.2f}%)")
print(f"  Legitimate: {(y_train_balanced==0).sum()} ({(y_train_balanced==0).sum()/len(y_train_balanced)*100:.2f}%)")

print("\nNow model will learn fraud patterns instead of ignoring them")

# visualize the rebalancing
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# before
y_train.value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Before Resampling')
axes[0].set_xlabel('Fraud (0=No, 1=Yes)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Legitimate', 'Fraudulent'], rotation=0)

# after
y_train_balanced.value_counts().plot(kind='bar', ax=axes[1], color=['green', 'red'])
axes[1].set_title('After Resampling')
axes[1].set_xlabel('Fraud (0=No, 1=Yes)')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(['Legitimate', 'Fraudulent'], rotation=0)

plt.tight_layout()
plt.savefig('../results/plots/03_class_imbalance_handling.png', dpi=100, bbox_inches='tight')
print("Saved visualization to results/plots/03_class_imbalance_handling.png")
plt.close()

print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)

print(f"""
Data preprocessing summary:
  - Loaded {df.shape[0]:,} transactions
  - Extracted time features (hour, day_of_week, month, is_weekend)
  - Calculated age from DOB
  - Handled missing values
  - Encoded categorical features (merchant, category, city, state, job)
  - Normalized all features
  - Split into 80% training, 20% testing
  - Resampled training data (fraud/legitimate now balanced)
  
Data is now ready in memory:
  - X_train_balanced: {X_train_balanced.shape[0]:,} balanced training features
  - y_train_balanced: {len(y_train_balanced)} balanced training labels
  - X_test_scaled: {X_test_scaled.shape[0]:,} test features
  - y_test: {len(y_test)} test labels

Key takeaways:
  1. Feature Engineering: Extracted useful info from dates/times
  2. Train-Test Split: Ensures unbiased performance evaluation
  3. Scaling: Makes features comparable
  4. Resampling: Handles class imbalance so model learns fraud patterns
  5. Test set should NOT be resampled (keep real-world distribution)
""")

print("="*80)