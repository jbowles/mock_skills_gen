import warnings
warnings.filterwarnings(action="ignore", message="^.*LAPACK bug 0038.*")
warnings.filterwarnings(action="ignore", message="^.*n_estimators.*")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

'''
Goal:
    Predict the Target for both the random and bias data sets
Specific:
    Find a good model... compare to dfb_model_one.py
    use one-hot encoding for the GenderLabel
    split general and specific atts to their own transform
    use normalization for general attrs
    use standardization for specific attrs
    use a median imputer for numerics
    and then introduce grid search
'''
# target_categories = [['Medicine', 1], ['GovBiz', 2], ['Education', 3], ['Arts', 4], ['Stem', 5]]

df = pd.read_csv('bias_student_metrics.csv')
# df = pd.read_csv('random_student_metrics.csv')

general_attr = ['Age', 'Social', 'Spatial', 'Temporal', 'Organizational', 'Verbal']
specific_attr = ['TechnicalWriting', 'DescriptiveWriting', 'AnalyticWriting', 'Arithmetic', 'Algebra', 'Geometry']
categorical_features = ['GenderLabel']
# numeric_features = general_attr+specific_attr

# ########### GENERAL DATA WRANGLING
# df[general_attr] = df[general_attr].astype(float)

# ########### test/train split 20/80 and DROP the TARGET we are trying to predict
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
train_drop_target = train_set.drop('Target', axis=1)
targets = train_set['Target'].copy()

# ########### FEATURE SCALING AND PREPARATION
# we've seen the models in dfb_model_one are still underfitting
# so now more messing with features with creative ways...
standardize_transform = Pipeline([('std_scaler', StandardScaler()), ('imputer', SimpleImputer(strategy='median'))])
normalize_transform = Pipeline([('mimax_scaler', MinMaxScaler()), ('imputer', SimpleImputer())])
cat_transform = Pipeline([('onehot', OneHotEncoder(categories='auto'))])
pre_proc = ColumnTransformer(
    [
        ('normalize', normalize_transform, general_attr),
        ('standardize', standardize_transform, specific_attr),
        ('category', cat_transform, categorical_features)
    ]
)
train_prepared = pre_proc.fit_transform(train_drop_target)


# ########### SOME DATA FOR VALIDATION
print("TrainingSet Size:", len(train_set))
print("")
some_data = train_drop_target.iloc[:10]
some_labels = targets.iloc[:10]
some_data_fitted = pre_proc.fit_transform(some_data)

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
#foreg = RandomForestRegressor()
foreg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                              max_features=6, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=800,
                              n_jobs=None, oob_score=False, random_state=None,
                              verbose=0, warm_start=False)

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
print("should have MSE: 0.42702668695878954")


# ########### GRID SEARCH
param_grid = [
    # {'n_estimators': [10, 100, 1000, 2000], 'max_features': [2, 4, 6, 8, 10]},
    {'n_estimators': [10, 100, 1000], 'max_features': [2, 4, 6, 8, 10]},
    {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]}
]

forest_reg = RandomForestRegressor()
forest_grid = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
forest_grid.fit(train_prepared, targets)
print("RandomForestRegression best_estimator_:", forest_grid.best_estimator_)
print("RandomForestRegression best_params_:", forest_grid.best_params_)
print("")


print("Fetching all evaluation scores...\n")
cvres = forest_grid.cv_results_
mean_squares = [np.sqrt(-x) for x in cvres['mean_test_score']]
cvres_params = cvres['params']
for mean_score, params in zip(mean_squares, cvres_params):
    print(mean_score, params)

print("best mean score:", np.min(mean_squares))
