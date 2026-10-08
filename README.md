# House Prices Advanced Regression Techniques

This repository contains my methodology for predicting continuous house prices based on historical housing data. 

## Methodology: Triple Blend Architecture

The final pipeline utilizes a weighted average of 3 distinct base algorithms. This configuration achieved a Root Mean Squared Logarithmic Error of 0.12663.

### 1. Data Engineering
Before feeding the data to the models, I performed strict mathematical transformations:
* Outlier Deletion: I dropped any house where the living area exceeded 4000 square feet to protect the dataset averages.
* Target Normalization: Because house prices are highly skewed, I applied a mathematical logarithm function to normalize the target distribution.
* Categorical Encoding: Text columns were converted into matrices of 0 and 1.

### 2. The 3 Base Models
I trained exactly 3 separate regression algorithms to capture different patterns in the housing data:
* XGBoost Regressor
* LightGBM Regressor
* Lasso Regression

### 3. Model Blending
I combined the predictions of the 3 base models using the following mathematical weights:
* 34 percent weight to XGBoost
* 33 percent weight to LightGBM
* 33 percent weight to Lasso

This simple voting system helps reduce overfitting and improves generalization to unseen test data.

## How to Run the Code
Navigate to the src directory and execute the python scripts.
Run the preprocessing script to clean the raw data and save it to the processed folder.
Then run the ensemble script to train the 3 models, blend their predictions, and generate the final submission file in the submissions folder.
