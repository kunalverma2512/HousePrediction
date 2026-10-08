import pandas as pd
import numpy as np

print("Loading raw data from the Datasets folder...")
train = pd.read_csv('../data/raw/train.csv')
test = pd.read_csv('../data/raw/test.csv')

print(f"Original Training shape: {train.shape}")
print(f"Original Testing shape: {test.shape}")

# BEST PRACTICE 1: Remove extreme outliers
outliers = train[train['GrLivArea'] > 4000].index
train = train.drop(outliers)
print(f"Dropped {len(outliers)} massive outlier houses.")

# BEST PRACTICE 2: Log transform the target variable
train['SalePrice'] = np.log1p(train['SalePrice'])
y_train = train['SalePrice']

train_features = train.drop(['Id', 'SalePrice'], axis=1)
test_features = test.drop(['Id'], axis=1)
all_data = pd.concat([train_features, test_features]).reset_index(drop=True)

# BEST PRACTICE 3: Smart Missing Value Imputation
none_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'MasVnrType']
for col in none_columns:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

zero_columns = ['GarageYrBlt', 'GarageArea', 'GarageCars', 
                'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
                'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_columns:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

# Impute LotFrontage with neighborhood median
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Fill remaining object columns with mode
object_cols = all_data.select_dtypes(include=['object']).columns
for col in object_cols:
    if all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Fill remaining numeric columns with median
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# BEST PRACTICE 4: Advanced Feature Engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBathrooms'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodelAge'] = all_data['YrSold'] - all_data['YearRemodAdd']

# BEST PRACTICE 5: Encoding categorical variables to numbers
all_data = pd.get_dummies(all_data)
print(f"Final Combined Feature shape after encoding: {all_data.shape}")

# Split back to train and test
X_train = all_data.iloc[:len(train_features), :]
X_test = all_data.iloc[len(train_features):, :]

# Save to the new organized folder
X_train_final = X_train.copy()
X_train_final['SalePrice'] = y_train.values
X_train_final.to_csv('../data/processed/cleaned_train.csv', index=False)
X_test.to_csv('../data/processed/cleaned_test.csv', index=False)
test[['Id']].to_csv('../data/processed/test_ids.csv', index=False)

print("Preprocessing complete! Cleaned files saved to project_code folder.")
