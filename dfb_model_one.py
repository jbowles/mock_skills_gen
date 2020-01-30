import warnings
warnings.filterwarnings(action="ignore", message="^.*LAPACK bug 0038.*")
warnings.filterwarnings(action="ignore", message="^.*n_estimators.*")
# https://github.com/scipy/scipy/issues/5998
# /usr/local/lib/python3.7/site-packages/sklearn/linear_model/base.py:503: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

'''
Goal:
    Predict the Target for both the random and bias data sets
Specific:
    Play with some models...
'''

df = pd.read_csv('bias_student_metrics.csv')
# df = pd.read_csv('random_student_metrics.csv')

# ########### test/train split 20/80 and DROP the TARGET we are trying to predict
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
train_drop_target = train_set.drop('Target', axis=1)
targets = train_set['Target'].copy()

# ########### FEATURE SCALING
#scaler = StandardScaler()
scaler = MinMaxScaler()
train_prepared = scaler.fit_transform(train_drop_target)

# ########### SOME DATA FOR VALIDATION
print("TrainingSet Size:", len(train_set))
print("")
some_data = train_drop_target.iloc[:10]
some_labels = targets.iloc[:10]
some_data_fitted = scaler.fit_transform(some_data)

# ########### LINEAR REGRESSION (based on what we saw in explore expect this to be ugly!)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(train_prepared, targets)

print("                     Targets:", list(some_labels))
print("LinearRegression Predictions:", [int(x) for x in linreg.predict(some_data_fitted)])
print("")

# ########### DECISION TREE REGRESSOR; let's see
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_prepared, targets)

print("                 Targets:", list(some_labels))
print("DecisionTree Predictions:", [int(x) for x in tree_reg.predict(some_data_fitted)])
print("")

# ########### RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
foreg = RandomForestRegressor()
foreg.fit(train_prepared, targets)

print("                 Targets:", list(some_labels))
print("RandomForest Predictions:", [int(x) for x in foreg.predict(some_data_fitted)])
print("")


# ########### MSE on linreg and tree_reg
from sklearn.metrics import mean_squared_error

linreg_predictions = linreg.predict(train_prepared)
linreg_MSE = mean_squared_error(targets, linreg_predictions)

tree_predictions = tree_reg.predict(train_prepared)
tree_MSE = mean_squared_error(targets, tree_predictions)

foreg_predictions = foreg.predict(train_prepared)
foreg_MSE = mean_squared_error(targets, foreg_predictions)


print("     LinearRegression MSE: ", np.sqrt(linreg_MSE))
print("DecisionTreeRegressor MSE: ", np.sqrt(tree_MSE))
print("RandomForestRegressor MSE: ", np.sqrt(foreg_MSE))
print("")


# ########### CROSS VALIDATION ..
from sklearn.model_selection import cross_val_score
treescores = cross_val_score(tree_reg, train_prepared, targets, scoring="neg_mean_squared_error", cv=10)
tree_MSE_scores = np.sqrt(-treescores)
linregscores = cross_val_score(linreg, train_prepared, targets, scoring="neg_mean_squared_error", cv=10)
linreg_MSE_scores = np.sqrt(-linregscores)
foregscores = cross_val_score(foreg, train_prepared, targets, scoring="neg_mean_squared_error", cv=10)
foreg_MSE_scores = np.sqrt(-foregscores)


print("CROSS VALIDATION SCORES.......\n")


def display_scores(scores):
    # print("Scores", scores)
    print("              Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print("")


print("===================cross_val_score linreg_Neg_MSE")
display_scores(linreg_MSE_scores)
print("===================cross_val_score tree_Neg_MSE")
display_scores(tree_MSE_scores)
print("===================cross_val_score foreg_Neg_MSE")
display_scores(foreg_MSE_scores)
