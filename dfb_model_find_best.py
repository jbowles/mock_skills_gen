import warnings
warnings.filterwarnings(action="ignore", message="^.*LAPACK bug 0038.*")
warnings.filterwarnings(action="ignore", message="^.*n_estimators.*")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from scipy import stats

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


def feature_scale(gattr, sattr, catfs, train_data):
    '''
    feature_scale(general_attr, specific_attr, categorical_features, train_drop_target)
    FEATURE SCALING AND PREPARATION
    we've seen the models in dfb_model_one are still underfitting
    so now more messing with features with creative ways...
    '''
    standardize_transform = Pipeline([('std_scaler', StandardScaler()), ('imputer', SimpleImputer(strategy='median'))])
    normalize_transform = Pipeline([('mimax_scaler', MinMaxScaler()), ('imputer', SimpleImputer())])
    cat_transform = Pipeline([('onehot', OneHotEncoder(categories='auto'))])
    pre_proc = ColumnTransformer(
        [
            ('normalize', normalize_transform, gattr),
            ('standardize', standardize_transform, sattr),
            ('category', cat_transform, catfs)
        ]
    )
    return (pre_proc.fit_transform(train_data), pre_proc)


def do_grid_search(nest, train_prep, targs):
    """
    forest_grid = do_grid_search(N_estimators, trained_prep, targets)
    """
    param_grid = [
        # {'n_estimators': [10, 100, 1000, 2000], 'max_features': [2, 4, 6, 8, 10]},
        {'n_estimators': nest, 'max_features': [6, 8]},
        {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]}
    ]

    forest_reg = RandomForestRegressor()
    forest_grid = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    forest_grid.fit(train_prep, targs)
    return forest_grid


def print_feature_importances(proc, grid, avail_attrs):
    '''
        print_feature_importances(pre_proc, forest_grid, general_attr+specific_attr)
    '''
    feature_importances = grid.best_estimator_.feature_importances_
    attrs = avail_attrs+list(proc.named_transformers_["category"][0].categories_[0])
    [print(x) for x in sorted(zip(feature_importances, attrs), reverse=True)]


def print_grid_search_results(grid):
    print("RandomForestRegression best_estimator_:", grid.best_estimator_)
    print("RandomForestRegression best_params_:", grid.best_params_)
    print("")
    print("Fetching all evaluation scores...\n")
    cvres = grid.cv_results_
    mean_squares = [np.sqrt(-x) for x in cvres['mean_test_score']]
    cvres_params = cvres['params']
    for mean_score, params in zip(mean_squares, cvres_params):
        print(mean_score, params)
    print("best mean score:", np.min(mean_squares))


def view_grid_feature_importance(proc, grid):
    """
    THIS CAN SHOW, ALONG WITH THE df.corr() CORRELATIONS THAT WE MAY BE ABLE
    TO DROP SOME FEATURES AND BY DOING SO SPEED UP TRAINING/INFERENCE
    """
    print_feature_importances(proc, grid, general_attr+specific_attr)
    print("best mean score:", np.min([np.sqrt(-x) for x in forest_grid.cv_results_['mean_test_score']]))
    print("RandomForestRegression best_params_:", grid.best_params_)
    print("RandomForestRegression best_estimator:", grid.best_estimator_)


def store_model(m, fname):
    '''
    best_model = joblib.load(joblib_file)
    '''
    joblib.dump(m, fname)


# ########### TEST INFERENCE
def test_inference(grid, proc):
    best_model = grid.best_estimator_

    X_test = test_set.drop('Target', axis=1)
    y_test = test_set['Target'].copy()

    # ######## DO NOT fit !!
    X_test_prepared = proc.transform(X_test)
    # ######## MAKE PREDICTIONS AND GET MSE,RMSE
    model_predictions = best_model.predict(X_test_prepared)
    final_MSE = mean_squared_error(y_test, model_predictions)
    final_RMSE = np.sqrt(final_MSE)
    print("final_MSE:", final_MSE)
    print("final_RMSE:", final_RMSE)
    return (model_predictions, X_test, y_test)


# ######## 95% CONFIDENCE INTERVAL...
def get_interval(pred, y):
    """
    95% interval
    get(model_predictions, y_test)
    stats.sem == Standard Error of Mean
    """
    squared_errors = (pred - y)**2
    return np.sqrt(
        stats.t.interval(
            0.95,
            len(squared_errors)-1,
            loc=squared_errors.mean(),
            scale=stats.sem(squared_errors)
        )
    )


train_prepared, pre_proc = feature_scale(general_attr, specific_attr, categorical_features, train_drop_target)
# N_estimators = [2, 6, 10]
# N_estimators = [200, 600, 1000]
N_estimators = [2000, 6000, 10000]
# forest_grid = do_grid_search(N_estimators, train_prepared, targets)
forest_grid = joblib.load('random_forest_regressor_model_1000.pkl')

if np.max(N_estimators) > 300:
    store_model(forest_grid, "random_forest_regressor_model_{}.pkl".format(np.max(N_estimators)))

# print_feature_importances(pre_proc, forest_grid, general_attr+specific_attr)
# print_grid_search_results(forest_grid)
# view_grid_feature_importance(pre_proc, forest_grid)

print("best mean score:", np.min([np.sqrt(-x) for x in forest_grid.cv_results_['mean_test_score']]))
predictions, X_test, y_test = test_inference(forest_grid, pre_proc)
print("95%% confidence interval:", get_interval(predictions, y_test))
