__author__ = 'Richard'
import pandas as pd
import numpy as np
import completed_project2_Util as util
import sys
import csv
import math
from sklearn.metrics import r2_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split

def read_data():
    df = pd.read_csv('kddcup98.csv')
    assert isinstance(df, pd.DataFrame) # for pycharm code completion
    #print df.head()
    #print df.describe()
    # remove duplicates
    df = df.drop_duplicates()
    df_alt = df['TARGET_D']
    # remove rows with dependent variable missing
    df = df.dropna(subset=['TARGET_B'])
    df = df.drop('TARGET_D', 1)
    return df, df_alt

def variable_selection(train, test, model):
    train_x, train_y = util.split_x_y(train.values)
    #test_x, test_y = util.split_x_y(test.values)

    train_vars = train.columns.tolist()

    rfecv = RFECV(model, step=1, cv=StratifiedKFold(train_y, 5),
            scoring='accuracy')

    selector = rfecv.fit(train_x, train_y)

    new_train_list = []
    count = 0
    for value in selector.support_:
        if value == True:
            new_train_list.append(train_vars[count])
            count += 1
    new_train_list.append('TARGET_B')

    return new_train_list

def linear_regression(train_X,train_y,test_X,test_y):

    num_folds = 5
    num_instances = len(train_X)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = {'n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9]}

    model = grid_search.GridSearchCV(linear_model.LinearRegression(), param_grid, cv=kfold, scoring='r2')
    model.fit(train_X, train_y)

    print "Linear Regression Model:"
    print model

    #print 'Coefficients: \n', regr.coef_
    pred_y = model.predict(test_X) # your predicted y values
    # The root mean square error
    mse = np.mean((pred_y - test_y) ** 2)
    rmse = math.sqrt(mse)
    print "Linear Regression Result:"
    print ("RMSE: %.2f" % rmse)
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

def knn_regression(train_X, train_y, test_X, test_y):
    num_folds = 5
    num_instances = len(train_X)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = [{'n_neighbors': [5, 6, 7, 8, 9, 10]}]
    knn = KNeighborsRegressor()

    model = grid_search.GridSearchCV(knn, param_grid, cv=kfold, scoring='r2')
    model.fit(train_X, train_y)
    print "KNN Regression Model:"
    print model
    #print 'Coefficients: \n', regr.coef_
    pred_y = model.predict(test_X) # your predicted y values
    # The root mean square error
    mse = np.mean((pred_y - test_y) ** 2)
    rmse = math.sqrt(mse)
    print ("RMSE: %.2f" % rmse)
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

def adaboost_regression(train_X, train_y, test_X, test_y):

    num_folds = 5
    num_instances = len(train_X)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = {'n_estimators': [50, 60, 70, 80, 90, 100]}
    boost = AdaBoostRegressor(loss='square')
    model = grid_search.GridSearchCV(boost, param_grid, cv=kfold, scoring='r2')

    print "AdaBoost Regression Model:"
    print model

    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    mse = np.mean((pred_y - test_y) ** 2)
    rmse = math.sqrt(mse)
    print "AdaBoost Regression Result:"
    print ("RMSE: %.2f" % rmse)
    r2 =  r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

def svm_regression(train_X, train_y, test_X, test_y):

    num_folds = 5
    num_instances = len(train_X)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = {'kernel': ['rbf', 'linear'], 'C': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    svm = SVR()

    model = grid_search.GridSearchCV(svm, param_grid, cv=kfold, scoring='r2')
    model.fit(train_X, train_y)

    print "SVM Regression Model:"
    print model

    pred_y = model.predict(test_X)
    mse = np.mean((pred_y - test_y) ** 2)
    rmse = math.sqrt(mse)
    print "SVM Regression Result:"
    print ("RMSE: %.2f" % rmse)
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

def random_forest_regression(train_X, train_y, test_X, test_y):

    num_folds = 5
    num_instances = len(train_X)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90]}
    forest = RandomForestRegressor()
    model = grid_search.GridSearchCV(forest, param_grid, cv=kfold, scoring='r2')

    model.fit(train_X, train_y)
    print "Random Forest Regression Model:"
    print model
    pred_y = model.predict(test_X)
    mse = np.mean((pred_y - test_y) ** 2)
    rmse = math.sqrt(mse)

    print "Random Forest Regression Result:"
    print ("RMSE: %.2f" % rmse)
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

def ridge_regression(train_X, train_y, test_X, test_y):

    num_folds = 5
    num_instances = len(train_X)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}]
    ridge = Ridge()
    model = grid_search.GridSearchCV(ridge, param_grid, cv=kfold, scoring='r2')

    model.fit(train_X, train_y)
    print "Ridge Regression Model:"
    print model
    #print 'Coefficients: \n', regr.coef_
    pred_y = model.predict(test_X) # your predicted y values
    # The root mean square error
    mse = np.mean((pred_y - test_y) ** 2)
    rmse = math.sqrt(mse)

    print "Ridge Regression Result:"
    print ("RMSE: %.2f" % rmse)
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

def fit_SGD_classifier(train_new, test_new):
    train_x, train_y = util.split_x_y(train_new.values)
    test_x, test_y = util.split_x_y(test_new.values)

    # make predictions
    num_folds = 5
    num_instances = len(train_x)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    model = grid_search.GridSearchCV(linear_model.SGDClassifier(loss='hinge', penalty='l2', learning_rate='optimal'), param_grid, cv=kfold, scoring='recall')
    model.fit(train_x, train_y)
    print "SGD Classifier Model:"
    print model

    expected = test_y
    predicted = model.predict(test_x)
    # summarize the fit of the model
    print "The SGD classifier results:"
    print(metrics.classification_report(expected, predicted))
    print "Accuracy Score:"
    print(metrics.accuracy_score(expected, predicted))
    #print(metrics.confusion_matrix(y, predicted))

def fit_random_forrest(train_new, test_new):
    train_x, train_y = util.split_x_y(train_new.values)
    test_x, test_y = util.split_x_y(test_new.values)

    num_folds = 5
    num_instances = len(train_x)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90]}
    model = grid_search.GridSearchCV(RandomForestClassifier(max_features='auto'), param_grid, cv=kfold, scoring='recall')
    model.fit(train_x, train_y)

    print "Random Forest Classifier Model:"
    print model
    expected = test_y
    predicted = model.predict(test_x)

    print "Random Forrest Classifier results:"
    print (metrics.classification_report(expected, predicted))
    print "Accuracy Score:"
    print (metrics.accuracy_score(expected, predicted))

def fit_adaboost(train_new, test_new):
    train_x, train_y = util.split_x_y(train_new.values)
    test_x, test_y = util.split_x_y(test_new.values)

    num_folds = 5
    num_instances = len(train_x)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = {'n_estimators': [50, 60, 70, 80, 90, 100]}
    model = grid_search.GridSearchCV(AdaBoostClassifier(), param_grid, cv=kfold, scoring='recall')
    model.fit(train_x, train_y)

    print "AdaBoost Classifier Model:"
    print model

    expected = test_y
    predicted = model.predict(test_x)

    print "AdaBoost Classifier results:"
    print (metrics.classification_report(expected, predicted))
    print "Accuracy Score:"
    print (metrics.accuracy_score(expected, predicted))

def fit_logistic_regression(train_new, test_new):
    train_x, train_y = util.split_x_y(train_new.values)
    test_x, test_y = util.split_x_y(test_new.values)

    num_folds = 5
    num_instances = len(train_x)
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds)
    param_grid = {'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = grid_search.GridSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_grid, cv=kfold, scoring='recall')
    model.fit(train_x, train_y)
    print "Logistic Regression Model:"
    print model

    expected = test_y
    predicted = model.predict(test_x)

    print "The logistic regression classification results:"
    print(metrics.classification_report(expected, predicted))
    print "Accuracy Score:"
    print(metrics.accuracy_score(expected, predicted))
    #print(metrics.confusion_matrix(expected, predicted))


def train_test_keep_some_vars(train, test, variables):
    train_new = train[variables]
    test_new = test[variables]
    return (train_new, test_new)

def split_train_test(arr, test_size=.3):
    train, test = train_test_split(arr, test_size=0.33)
    train_X = train[:, :-1]
    #print train_X.shape
    train_y = train[:, -1]
    #print train_y.shape
    test_X = test[:,:-1]
    test_y = test[:, -1]
    return train_X,train_y,test_X,test_y

def missing_imputation_for_numeric(df, numeric_with_na):
    for var in numeric_with_na:
        if "log_" in var or "sqrt_" in var:
            util.process_missing_numeric_no_dummy(df, var)
        else:
            util.process_missing_numeric_with_dummy(df, var)
    return

def create_categorical_dummy(df, categorical):
    for item in categorical:
        dummy_var = pd.get_dummies(df[item], prefix=item).sort_index()
        for i in dummy_var.columns.tolist():
            df.insert(df.columns.get_loc(item), i, dummy_var[i])

    return

def main():
    #Task 1:

    # Step 1. Import data
    df, df_alt = read_data()
    print df.head()
    print df.describe()
    # Step 2. Explore data
    # 2.1. Get variable names
    col_names = df.columns.tolist()
    print col_names
    # 2.2. Classify variables into numeric, categorical (with strings), and nominal
    df['DemMedHomeValue'] = df['DemMedHomeValue'].replace('[\$,)]','', regex=True).astype(float)
    df['DemMedIncome'] = df['DemMedIncome'].replace('[\$,)]','', regex=True).astype(float)
    numeric,categorical,nominal = util.variable_type(df)
    print "numeric:", numeric
    print "categorical:", categorical
    print "nominal:", nominal
    # 2.3. Draw histogram for numeric variables
    util.draw_histograms(df, ['ID', 'GiftCnt36', 'GiftCntAll', 'GiftCntCard36', 'GiftCntCardAll', 'GiftAvgLast', 'GiftAvg36', 'GiftAvgAll', 'GiftAvgCard36'], 3, 3)
    util.draw_histograms(df, ['GiftTimeLast', 'GiftTimeFirst', 'PromCnt12', 'PromCnt36', 'PromCntAll', 'PromCntCard12', 'PromCntCard36', 'PromCntCardAll', 'DemCluster'], 3, 3)
    util.draw_histograms(df, ['DemAge', 'DemPctVeterans', 'DemMedHomeValue', 'DemMedIncome'], 2, 2)
    # 2.4. Quality Check
    num_var_with_dollar_zero = ['DemMedHomeValue', 'DemMedIncome']

    for val in num_var_with_dollar_zero:
        util.replace_missing_dollar(df, val)

    numeric_variables_with_na = util.variables_with_missing(df)
    #print numeric_variables_with_na

    for value in numeric_variables_with_na:
        util.replace_with_missing(df, value)
    # 2.5. Draw pie charts for categorical variables
    util.draw_piecharts(df, ['StatusCat96NK', 'DemGender', 'DemHomeOwner'], 1,3)
    # 2.6. Identify variables that have skewed distribution and need to be log or sqrt-transformed
    variables_needs_tranform = ['GiftCnt36', 'GiftCntAll', 'GiftCntCard36', 'GiftCntCardAll', 'GiftAvgLast', 'GiftAvg36', 'GiftAvgAll', 'GiftAvgCard36', 'GiftTimeLast', 'GiftTimeFirst', 'PromCnt12', 'PromCnt36', 'PromCntAll', 'PromCntCard12', 'PromCntCard36', 'PromCntCardAll', 'DemCluster', 'DemMedHomeValue', 'DemMedIncome']
    # Step 3. Transform Variables
    # 3.1 Categorical Missing Value Imputation
    create_categorical_dummy(df, categorical)
    df = df.drop('DemGender', 1)
    df = df.drop('DemHomeOwner', 1)
    df = df.drop('StatusCat96NK', 1)
    # 3.2 Missing Value Imputation:
    missing_imputation_for_numeric(df, numeric_variables_with_na) # do missing value imputation
    # 3.3 Log Transformation for Continuous Variables
    for value in variables_needs_tranform:
        util.add_log_transform(df, value)
    # Step 4. Data Partitioning
    # 4.1 Clean Data:
    #print df.columns.tolist()
    vars = ['GiftCnt36', 'log_GiftCnt36', 'GiftCntAll', 'log_GiftCntAll', 'GiftCntCard36', 'log_GiftCntCard36', 'GiftCntCardAll', 'log_GiftCntCardAll', 'GiftAvgLast', 'log_GiftAvgLast', 'GiftAvg36', 'log_GiftAvg36', 'GiftAvgAll', 'log_GiftAvgAll', 'GiftAvgCard36', 'log_GiftAvgCard36', 'GiftTimeLast', 'log_GiftTimeLast', 'GiftTimeFirst', 'log_GiftTimeFirst', 'PromCnt12', 'log_PromCnt12', 'PromCnt36', 'log_PromCnt36', 'PromCntAll', 'log_PromCntAll', 'PromCntCard12', 'log_PromCntCard12', 'PromCntCard36', 'log_PromCntCard36', 'PromCntCardAll', 'log_PromCntCardAll', 'StatusCat96NK_A', 'StatusCat96NK_E', 'StatusCat96NK_F', 'StatusCat96NK_L', 'StatusCat96NK_N', 'StatusCat96NK_S', 'StatusCatStarAll', 'DemCluster', 'log_DemCluster', 'DemAge', 'DemGender_F', 'DemGender_M', 'DemGender_U', 'DemHomeOwner_H', 'DemHomeOwner_U', 'DemMedHomeValue', 'log_DemMedHomeValue', 'DemPctVeterans', 'DemMedIncome', 'log_DemMedIncome', 'GiftCntAll_missing', 'GiftAvgCard36_missing', 'DemAge_missing', 'DemMedHomeValue_missing', 'DemMedIncome_missing', 'TARGET_B']
    df = df[vars]
    # 4.2 Split Data in to Training and Test
    train, test = util.split_train_test_frame(df, test_size=.4)
    # Step 5 RFECV Variable Selection
    model = LogisticRegression(class_weight='balanced')
    selected_vars = variable_selection(train, test, model)
    print "Selected Variables:"
    print selected_vars
    # Step 6 Model Fitting and Comparison
    train_new, test_new= train_test_keep_some_vars(train, test, selected_vars)
    # 6.1 Logistic Regression
    fit_logistic_regression(train_new, test_new)
    # 6.2 SGD Classifier
    fit_SGD_classifier(train_new, test_new)
    # 6.3 Random Forrest Classifier
    #fit_random_forrest(train_new, test_new)
    # 6.4 AdaBoost Classifier
    fit_adaboost(train_new, test_new)

    # Task 2: Regression
    #Step 1: Remove Target_B, add TARGET_D and remove rows with missing
    df2 = pd.concat([df, df_alt], axis=1) #df_alt was obtained in step one of Task 1 (TARGET_D)
    df2 = df2.drop('TARGET_B', 1)
    df2 = df2.dropna()
    df2 = df2.drop('index', 1)
    #print df2.head()
    #Step2: Apply changes to the selected variables in Step 5
    selected_vars.remove('TARGET_B')
    selected_vars.append('TARGET_D')
    print "Selected Variables:"
    print selected_vars
    #Step3: Implement new dataframe with selected variables including TARGET_D
    df2 = df2[selected_vars]
    arr = df2.values
    #Step4: Split data in to training and test for regression models
    train_X,train_Y,test_X,test_Y = split_train_test(arr)
    #Step5: Fit regression models
    #5.1: Ridge Regression
    ridge_regression(train_X, train_Y, test_X, test_Y)
    #5.2: Linear Regression
    linear_regression(train_X, train_Y, test_X, test_Y)
    #5.3: AdaBoost Regression
    #adaboost_regression(train_X, train_Y, test_X, test_Y)
    #5.4: SVM Regression
    svm_regression(train_X, train_Y, test_X, test_Y)
    #5.5: Random Forest Regression
    #random_forest_regression(train_X, train_Y, test_X, test_Y)
    #5.6: KNN Regression
    #knn_regression(train_X, train_Y, test_X, test_Y)


if __name__ == "__main__":
    main()