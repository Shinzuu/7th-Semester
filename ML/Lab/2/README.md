# ML Lab 2: Data Preprocessing - Flood Prediction

## Project Overview

**Dataset:** Kaggle Playground Series S4E5 - Flood Prediction
**Task:** Predict FloodProbability using 20 flood-related features
**Dataset Size:** 1,117,957 samples

## Folder Structure

```
ML/Lab/2/
├── code/
│   ├── preprocessing.ipynb          # Complete preprocessing notebook
│   └── correlation_heatmap.png      # Feature correlation visualization
├── data/
│   ├── train_original.csv           # Original training data (backup)
│   ├── test_original.csv            # Original test data (backup)
│   ├── train_preprocessed.csv       # Ready for model training (670,773 rows)
│   ├── valid_preprocessed.csv       # Ready for validation (223,592 rows)
│   └── test_preprocessed.csv        # Ready for testing (223,592 rows)
├── playground-series-s4e5/          # Original Kaggle dataset
├── preprocessing_steps_complete.md  # Reference guide (Part 1 & 2)
├── venv/                            # Python virtual environment
└── README.md                        # This file
```

## Preprocessing Summary

### 1. Data Exploration ✓
- **Rows:** 1,117,957
- **Columns:** 22 (20 features + 1 ID + 1 target)
- **Target:** FloodProbability (continuous, range: 0.285 - 0.725)
- **Features:** All numerical (integer scores 0-18)

### 2. Data Cleaning ✓
- ✅ **No missing values** found
- ✅ **No duplicate rows** found
- ✅ **Outliers checked:** All values are valid domain scores

### 3. Feature Selection ✓
- **Correlation Analysis:** No multicollinearity detected
- **All 20 features retained** - all contribute similarly to target
- **Feature-Target Correlation:** Weak-to-moderate (0.17-0.19)

### 4. Data Splitting ✓
- **Training Set:** 670,773 samples (60%)
- **Validation Set:** 223,592 samples (20%)
- **Test Set:** 223,592 samples (20%)
- **Random State:** 42 (for reproducibility)

### 5. Feature Scaling ✓
- **Method:** StandardScaler (Mean=0, Std=1)
- **Fit on:** Training data only
- **Transform:** Validation and test sets
- **Result:** All features normalized to standard normal distribution

## Files Generated

### Preprocessed Data (Ready for ML):
1. `train_preprocessed.csv` - For model training
2. `valid_preprocessed.csv` - For hyperparameter tuning
3. `test_preprocessed.csv` - For final model evaluation

### Visualization:
- `correlation_heatmap.png` - Feature correlation matrix

### Reference Documents:
- `preprocessing_steps_complete.md` - Complete preprocessing guide
- `preprocessing.ipynb` - Jupyter notebook with all steps

## Features (20 Total)

1. MonsoonIntensity
2. TopographyDrainage
3. RiverManagement
4. Deforestation
5. Urbanization
6. ClimateChange
7. DamsQuality
8. Siltation
9. AgriculturalPractices
10. Encroachments
11. IneffectiveDisasterPreparedness
12. DrainageSystems
13. CoastalVulnerability
14. Landslides
15. Watersheds
16. DeterioratingInfrastructure
17. PopulationScore
18. WetlandLoss
19. InadequatePlanning
20. PoliticalFactors

## How to Use

### 1. Open Jupyter Notebook
```bash
cd ML/Lab/2/code
source ../venv/bin/activate
jupyter notebook preprocessing.ipynb
```

### 2. Run All Cells
Execute all cells in order to reproduce the preprocessing steps.

### 3. Load Preprocessed Data (For Modeling)
```python
import pandas as pd

# Load preprocessed data
train = pd.read_csv('../data/train_preprocessed.csv')
valid = pd.read_csv('../data/valid_preprocessed.csv')
test = pd.read_csv('../data/test_preprocessed.csv')

# Separate features and target
X_train = train.drop(['id', 'FloodProbability'], axis=1)
y_train = train['FloodProbability']

X_valid = valid.drop(['id', 'FloodProbability'], axis=1)
y_valid = valid['FloodProbability']

X_test = test.drop(['id', 'FloodProbability'], axis=1)
y_test = test['FloodProbability']
```

## Next Steps

1. **Model Selection:** Choose appropriate regression models
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting
   - Neural Networks

2. **Model Training:** Train on `train_preprocessed.csv`

3. **Hyperparameter Tuning:** Optimize using `valid_preprocessed.csv`

4. **Final Evaluation:** Test on `test_preprocessed.csv`

5. **Metrics to Track:**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score

## Environment Setup

### Virtual Environment (Already Created)
```bash
# Activate environment
source venv/bin/activate

# Deactivate when done
deactivate
```

### Installed Packages
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter
- notebook

## Notes for Google Colab

When transferring to Google Colab:

1. Upload `train_preprocessed.csv`, `valid_preprocessed.csv`, `test_preprocessed.csv`
2. Copy all cells from `preprocessing.ipynb`
3. Install required packages:
```python
!pip install pandas numpy matplotlib seaborn scikit-learn
```

## Key Insights

1. **Clean Dataset:** No missing values or duplicates - high quality data
2. **No Multicollinearity:** All features are independent
3. **Similar Feature Importance:** All features contribute equally (correlation 0.17-0.19)
4. **Balanced Approach:** All preprocessing steps follow best practices from reference guide

## References

- Kaggle Competition: [Playground Series S4E5](https://www.kaggle.com/competitions/playground-series-s4e5)
- Preprocessing Guide: `preprocessing_steps_complete.md`
- Medium Article: Data Preprocessing Steps for Machine Learning (Part 1 & 2)

---

**Created:** January 8, 2026
**Lab:** ML Lab 2 - Data Preprocessing
**Status:** ✅ Complete - Ready for Model Training
