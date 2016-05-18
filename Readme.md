# Python Project 2 - Data Mining

> In this project, we will do data mining on a relatively large data set - 1998 KDD CUP

> There are two target variables in the dataset, TARGET_B and TARGET_D.
> TARGET_B is binary, representing whether a person donated or not.
> TARGET_D is continuous, representing how much a person donated.

1. Imported Data and drop TARGET_D

2. Explore Data
- Get Variable Names
- Classify variables as numeric, categorical or nominal.
- Draw Histograms for numeric variables
- Quality Check: Replace false or unreasonable variables with np.nan.
- Identify numeric variables with skewed distributions that need to be log transformed.
- Draw pie charts for categorical variables

3. Tranform Variables
- Do categorical/nominal variable recoding and missing value imputation using dummy variables.
- For continuous variables with skewed distributions, do log transformation
- Do missing value imputation for continuous variables

4. Data partitioning (60% training and 40% test)

5. Variable selection using RFECV (recursive feature elimination with cross-validation)

6. Model fitting and comparison
- Use Logistic Regression as baseline model
- Use SGD Classifier with L2 norm regularization and grid search to set optimal alpha
- Use AdaBoost Classifier with grid search
- Print classification report and accuracy scores for each model

7. In part two, we remove rows with missing TARGET_D values and drop TARGET_B.

8. Fit models
- Fit Ridge Regression model and print RMSE and R2 values
- Fit Linear Regression model and print RMSE and R2 values
- Fit SVM Regression model and print RMSE and R2 values