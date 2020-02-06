import warnings
warnings.filterwarnings(action="ignore", message="^.*LAPACK bug 0038.*")
warnings.filterwarnings(action="ignore", message="^.*n_estimators.*")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

'''
Goal:
    Predict the Target for both the random and bias data sets
Specific:
    Find a good model...
'''
# target_categories = [['Medicine', 1], ['GovBiz', 2], ['Education', 3], ['Arts', 4], ['Stem', 5]]
general_attr = ['Age', 'Social', 'Spatial', 'Temporal', 'Organizational', 'Verbal']
specific_attr = ['TechnicalWriting', 'DescriptiveWriting', 'AnalyticWriting', 'Arithmetic', 'Algebra', 'Geometry']
categorical_features = ['GenderLabel']

df = pd.read_csv('bias_student_metrics.csv')
# df = pd.read_csv('random_student_metrics.csv')

# ######### NOW DO CLASSIFICATION !!!!!!!!!!!!!!!1...
X = df.drop('Target', axis=1)
y = df['Target'].copy()
# X=features, y=labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

standardize_transform = Pipeline([('std_scaler', StandardScaler()), ('imputer', SimpleImputer(strategy='median'))])
normalize_transform = Pipeline([('mimax_scaler', MinMaxScaler()), ('imputer', SimpleImputer())])
cat_transform = Pipeline([('onehot', OneHotEncoder(categories='auto'))])

preprocessor = ColumnTransformer(
    [
        ('normalize', normalize_transform, general_attr),
        ('standardize', standardize_transform, specific_attr),
        ('category', cat_transform, categorical_features),
    ]
)

iters = 10000
# top_grid_socre==0.9050, clf_fit_score==0.9038
sol_ver = 'lbfgs'
lnorms = ['l2']
multiC = 'multinomial'
#iters = 3000
# top_grid_socre==0.8798, clf_fit_score==0.8745
# sol_ver = 'liblinear'
# lnorms = ['l2', 'l1']
clf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver=sol_ver, max_iter=iters, multi_class=multiC))
    ]
)
clf.fit(X_train, y_train)
clf_fit_score = clf.score(X_test, y_test)
print("model score accuracy: %.3f (should be around 90%% already!!!!)" % clf_fit_score)

param_grid = [
    {
        'classifier': [LogisticRegression()],
        'classifier__penalty': lnorms,
        # 'classifier__C': np.logspace(-4, 4, 20),
        'classifier__C': [0.1, 1.0, 10, 100],
        'classifier__solver': [sol_ver],
        'classifier__multi_class': [multiC],
        'classifier__max_iter': [iters],
    },
]
logreg = Pipeline([('classifier', LogisticRegression(solver=sol_ver, max_iter=iters, multi_class=multiC))])
grid_search = GridSearchCV(logreg, param_grid, cv=5, verbose=True)
grid_search.fit(X_train, y_train)
top_grid_score = grid_search.score(X_test, y_test)
print("best LogistRegression accuracy from grid search: %.3f" % top_grid_score)
print("did classifier get us best score already? top_grid_socre==%.4f, clf_fit_score==%.4f" % (top_grid_score, clf_fit_score))

# ########### WHAT NEXT?? ROC and AUC... F1... then move on to RandomForest for LogisticRegression!!!!

# ############### GRID SEARCH for RandomForest LogisticRegression
# param_grid = [
#    {'classifier': [LogisticRegression()],
#     'classifier__penalty': ['l1', 'l2'],
#     'classifier__C': np.logspace(-4, 4, 20),
#     'classifier__solver': ['lbfgs']},
#    {'classifier': [RandomForestClassifier()],
#     'classifier__n_estimators': list(range(10, 101, 10)),
#     'classifier__max_features': list(range(6, 32, 5))}
# ]
