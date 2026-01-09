# Complete Data Preprocessing Steps for Machine Learning
## Reference Guide for ML Lab 2

Source: Data Preprocessing Steps for Machine Learning in Python (Part 1 & 2) by Learn with Nas

---

## Table of Contents
1. [Data Collection](#step-1-data-collection)
2. [Data Cleaning](#step-2-data-cleaning)
3. [Data Transformation](#step-3-data-transformation)
4. [Feature Engineering: Scaling, Normalization, Standardization](#step-4-feature-engineering)
5. [Feature Selection](#step-5-feature-selection)
6. [Handling Imbalanced Data](#step-6-handling-imbalanced-data)
7. [Encoding Categorical Features](#step-7-encoding-categorical-features)
8. [Data Splitting](#step-8-data-splitting)

---

## Step 1: Data Collection

**Purpose:** Gather data aligned with project goals and objectives

**Key Points:**
- Quality data is essential for model success
- Sources: Kaggle, DataHub, Google Data Search, CKAN, Quandl, etc.

**Implementation:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('data/dataset.csv')

# Initial exploration
df.head()
df.info()
df.describe()
```

---

## Step 2: Data Cleaning

### 2a. Handling Missing Values

**Process:**

1. **Identify missing values**
```python
# Check dataset info
df.info()

# Check percentage of missing values
df.isna().sum() / len(df)
```

2. **Visualize missing data**
```python
plt.figure(figsize=(10,6))
sns.heatmap(df.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
plt.title('Missing Data Heatmap')
plt.show()
```

3. **Fill missing values**
```python
# Imputation function using median (prevents outliers)
def data_imputation(data, column_grouping, column_selected):
    """
    Fill missing values using median grouped by category

    Parameters:
    - data: DataFrame to process
    - column_grouping: Column used to group values
    - column_selected: Column to fill NaN values
    """
    group = data[column_grouping].unique()

    for value in group:
        # Get median
        median = data.loc[(data[column_grouping]==value) &
                         ~(data[column_selected].isna()),
                         column_selected].median()

        # Fill missing values
        data.loc[(data[column_grouping]==value) &
                (data[column_selected].isna()),
                column_selected] = median

    return data

# Apply imputation
df = data_imputation(data=df, column_grouping='category', column_selected='target_column')
```

**Key Decision:** Median is preferred over mean to avoid outlier influence

---

### 2b. Handling Outliers

**Process:**

1. **Visualize outliers**
```python
# Boxplot for outlier detection
sns.boxplot(df['column_name'])
plt.title('Outlier Detection')
plt.show()

# Check statistics
df['column_name'].describe()
```

2. **Handle outliers**
```python
# Remove negative signs (if incorrect)
df['column'] = abs(df['column'])

# Round decimal places
df['column'] = round(df['column'], 0)

# Replace outliers with median
threshold = 200000  # Define based on domain knowledge
condition = (df['column'] > threshold) & (df['column'].notnull())
df['column'] = df['column'].mask(condition, df['column'].median())
```

3. **Verify results**
```python
sns.boxplot(df['column'])
plt.title('After Outlier Treatment')
plt.show()
```

**Key Points:**
- Use domain knowledge to set thresholds
- Median replacement prevents new outliers
- Always verify with visualization

---

### 2c. Handling Duplicates

**Process:**
```python
# Check for duplicates
print(f"Number of duplicates: {df.duplicated().sum()}")

# Remove duplicates and reset index
df = df.drop_duplicates().reset_index(drop=True)

# Verify
print(f"Shape after removing duplicates: {df.shape}")
```

---

## Step 3: Data Transformation

**Purpose:** Convert data format for better analysis without changing content

### 3a. Using groupby()

```python
# Group and aggregate data
grouped_df = df.groupby(['column1', 'column2'])[['value1', 'value2']].sum()

# With filtering
filtered_df = (df.groupby(['platform', 'name'])[['sales', 'score']]
               .sum()
               .query('platform == "PS3" & score > 0')
               .reset_index())
```

### 3b. Using pivot_table()

```python
# Create pivot table for multidimensional summary
pivot_df = pd.pivot_table(data=df,
                          index='category',
                          values=['sales_region1', 'sales_region2'],
                          aggfunc='sum')

# Visualize aggregated data
plt.figure(figsize=(20,6))
plt.title('Sales Distribution Across Regions')
sns.lineplot(data=pivot_df)
plt.show()
```

**Key Points:**
- GroupBy: One-dimensional aggregation
- Pivot Table: Two-dimensional summary
- Useful for exploring relationships

---

## Step 4: Feature Engineering: Scaling, Normalization, Standardization

**Purpose:** Standardize feature values to uniform scale

**When to Use:**
- Gradient descent algorithms (Linear/Logistic Regression)
- Distance-based algorithms (KNN, K-means, SVM)
- Neural networks

### 4a. Feature Scaling (MaxAbsScaler)

```python
from sklearn.preprocessing import MaxAbsScaler

# Scale to maximum value (range: -1 to 1)
feature_names = ['age', 'income', 'score']
transformer = MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer.transform(df[feature_names].to_numpy())
```

### 4b. Normalization (MinMaxScaler)

```python
from sklearn.preprocessing import MinMaxScaler

# Scale to range [0, 1]
scaler = MinMaxScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)
```

**Important:** Always fit on training data, then transform test data

### 4c. Standardization (StandardScaler)

```python
from sklearn.preprocessing import StandardScaler

# Standardize to mean=0, std=1
scaler = StandardScaler()
numerical_cols = ['age', 'income', 'balance']

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_valid[numerical_cols] = scaler.transform(X_valid[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
```

**Key Points:**
- StandardScaler: Mean=0, Std=1 (normal distribution)
- MinMaxScaler: Range [0,1]
- MaxAbsScaler: Range [-1,1]
- Improves model performance (F1, AUC-ROC)

---

## Step 5: Feature Selection

**Purpose:** Select optimal features that influence model performance

### Categories:
- **Supervised Techniques:** For labeled data (Linear Regression, Decision Trees, SVM)
- **Unsupervised Techniques:** For unlabeled data (K-Means, PCA, Hierarchical Clustering)

### Methods:
1. **Filter Methods:** Fast, use univariate statistics, good for high-dimensional data
2. **Wrapper Methods:** Use ML algorithm, exhaustive search, better accuracy but slower
3. **Embedded Methods:** Combine advantages of both, iterative selection

### Correlation Coefficient Technique

**Detect multicollinearity:**
```python
# Correlation heatmap
plt.figure(figsize=(20,15))
heatmap = sns.heatmap(df.corr(), annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12})
plt.show()

# Remove highly correlated features (correlation > 0.9)
# Example: If feature A and B have correlation 0.96, remove one
df = df.drop('highly_correlated_feature', axis=1)
```

**Select features based on target correlation:**
```python
# Check correlation with target
df.corr()['target'].sort_values()

# Select features with correlation > 0.1 or < -0.1
selected_features = df.corr()['target'][
    (df.corr()['target'] > 0.1) | (df.corr()['target'] < -0.1)
].index.tolist()
```

### Detect Anomalous Data with KNN

```python
from pyod.models.knn import KNN

# Select features for outlier detection
outliers = df[selected_features]

# Fit KNN model
model = KNN()
model.fit(outliers)

# Predict outliers
outliers['is_outlier'] = model.predict(outliers) == 1
anomalies_count = outliers['is_outlier'].sum()
print(f"Number of Anomalies: {anomalies_count}")

# Remove anomalies
outlier_keys = list(outliers[outliers['is_outlier'] == 1].index)
df_clean = df.drop(outlier_keys)
```

**Key Points:**
- Remove features with very high correlation (>0.9) to avoid multicollinearity
- Select features with meaningful correlation to target
- Use outlier detection to remove anomalous data

---

## Step 6: Handling Imbalanced Data

**Purpose:** Address uneven distribution in target class

**Problem:** One class has significantly more observations than another

### Approaches:

#### 6a. Choose Proper Evaluation Metric
- Don't rely solely on accuracy
- Use: Precision, Recall, F1-Score, AUC-ROC

#### 6b. Upsampling (Oversampling)

```python
from sklearn.utils import shuffle

def upsample(features, target, repeat):
    """
    Increase minority class frequency

    Parameters:
    - features: Feature DataFrame
    - target: Target Series
    - repeat: How many times to repeat minority class
    """
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    # Repeat minority class
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    # Shuffle
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=42
    )

    return features_upsampled, target_upsampled

# Apply upsampling
features_train_up, target_train_up = upsample(features_train, target_train, repeat=4)

# Check balance
print(target_train_up.value_counts())
```

#### 6c. Downsampling (Undersampling)

```python
def downsample(features, target, fraction):
    """
    Decrease majority class frequency

    Parameters:
    - features: Feature DataFrame
    - target: Target Series
    - fraction: Fraction of majority class to keep
    """
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    # Sample majority class
    features_downsampled = pd.concat([
        features_zeros.sample(frac=fraction, random_state=42),
        features_ones
    ])
    target_downsampled = pd.concat([
        target_zeros.sample(frac=fraction, random_state=42),
        target_ones
    ])

    # Shuffle
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=42
    )

    return features_downsampled, target_downsampled

# Apply downsampling
features_train_down, target_train_down = downsample(features_train, target_train, fraction=0.3)

# Check balance
print(target_train_down.value_counts())
```

#### 6d. SMOTE (Synthetic Minority Oversampling Technique)

```python
from imblearn.over_sampling import SMOTE

# Create synthetic samples for minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check balance
print(y_train_smote.value_counts())
```

**Key Points:**
- Upsampling: Increases minority class (may introduce synthetic data)
- Downsampling: Decreases majority class (may lose information)
- SMOTE: Creates synthetic samples using k-nearest neighbors
- Always check class distribution before and after

---

## Step 7: Encoding Categorical Features

**Purpose:** Convert categorical data to numerical format for ML algorithms

### 7a. Ordinal Encoding

**Use when:** Categories have inherent order (e.g., 'bad', 'average', 'good')

```python
from sklearn.preprocessing import OrdinalEncoder

# Define order
categories = [['bad', 'average', 'good']]
encoder = OrdinalEncoder(categories=categories)

df['quality_encoded'] = encoder.fit_transform(df[['quality']])
# Result: bad=0, average=1, good=2
```

### 7b. Nominal Encoding (One-Hot Encoding)

**Use when:** Categories have NO inherent order (e.g., 'red', 'blue', 'green')

```python
# Method 1: Using pandas
categorical_cols = ['geography', 'gender']
df_encoded = pd.get_dummies(df, drop_first=True, columns=categorical_cols)

# Method 2: Using sklearn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(df[categorical_cols])

# Create DataFrame with encoded columns
encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder.get_feature_names_out(categorical_cols)
)
```

**Key Points:**
- **Ordinal:** Preserves order (0, 1, 2, ...)
- **Nominal:** Creates binary columns (0 or 1)
- `drop_first=True`: Avoids multicollinearity (drops one category)
- One-hot encoding increases number of features

---

## Step 8: Data Splitting

**Purpose:** Partition dataset for training, validation, and testing

### Standard Split Ratios:
- **Training Set:** 60-80% (model learns patterns)
- **Validation Set:** 10-20% (tune hyperparameters, select model)
- **Test Set:** 10-20% (final evaluation)

### Implementation:

```python
from sklearn.model_selection import train_test_split

# Split into train+validation (80%) and test (20%)
df_train_valid, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Split train+validation into train (60%) and validation (20%)
df_train, df_valid = train_test_split(df_train_valid, test_size=0.25, random_state=42)
# 0.25 of 80% = 20%

# Separate features and target for training set
features_train = df_train.drop('target_column', axis=1)
target_train = df_train['target_column']

# Separate features and target for validation set
features_valid = df_valid.drop('target_column', axis=1)
target_valid = df_valid['target_column']

# Separate features and target for test set
features_test = df_test.drop('target_column', axis=1)
target_test = df_test['target_column']

# Verify shapes
print(f"Training set: {features_train.shape}")
print(f"Validation set: {features_valid.shape}")
print(f"Test set: {features_test.shape}")
```

### Important Notes:

**Training Set:**
- Used to train the model
- Should be representative of all classes
- Must be high quality and unbiased

**Validation Set:**
- Used for hyperparameter tuning
- Helps select best model/parameters
- Prevents overfitting to training data
- Use for model comparison

**Test Set:**
- **ONLY use once** for final evaluation
- Do NOT use for model selection
- Provides unbiased performance estimate
- Represents real-world performance

**Key Points:**
- Always use `random_state` for reproducibility
- Never train on validation/test data
- Never tune on test data
- Test set is the final checkpoint

---

## Complete Preprocessing Workflow

```python
# 1. Load Data
df = pd.read_csv('data.csv')

# 2. Data Cleaning
# - Handle missing values
# - Handle outliers
# - Remove duplicates

# 3. Feature Engineering
# - Scale/Normalize/Standardize

# 4. Feature Selection
# - Remove highly correlated features
# - Select relevant features

# 5. Handle Imbalanced Data (if applicable)
# - Upsample/Downsample/SMOTE

# 6. Encode Categorical Features
# - Ordinal or One-Hot encoding

# 7. Split Data
# - Train/Validation/Test

# 8. Train Model
# - Use training set

# 9. Tune Model
# - Use validation set

# 10. Final Evaluation
# - Use test set (ONCE)
```

---

## Essential Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.utils import shuffle

from pyod.models.knn import KNN
from imblearn.over_sampling import SMOTE
```

---

## Tips for Your Project

1. **Always visualize** before and after preprocessing
2. **Document decisions** (why median vs mean, why remove vs replace)
3. **Check distributions** with `.describe()`
4. **Use appropriate scaling** for your ML algorithm
5. **Keep original data** copy before transformations
6. **Reset index** after major operations
7. **Set random_state** for reproducibility
8. **Never look at test data** until final evaluation
9. **Verify each step** with shape checks and visualizations
10. **Handle imbalanced data** before splitting or after splitting (but before training)

---

## Common Pitfalls to Avoid

❌ Fitting scaler on entire dataset (causes data leakage)
✅ Fit on training data, transform validation/test

❌ Using test data for model selection
✅ Use validation data for selection, test for final eval

❌ Dropping too many features
✅ Use correlation and domain knowledge

❌ Ignoring class imbalance
✅ Check distribution and handle if needed

❌ Encoding before splitting (causes data leakage)
✅ Split first, then encode (or use pipelines)

❌ Not setting random_state
✅ Always set for reproducibility

---

**End of Preprocessing Guide**

*Good luck with your ML Lab 2 project!*
