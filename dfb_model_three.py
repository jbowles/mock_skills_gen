import warnings
warnings.filterwarnings(action="ignore", message="^.*LAPACK bug 0038.*")
warnings.filterwarnings(action="ignore", message="^.*n_estimators.*")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt

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

print("")
print("##########Running GridSearch############")
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

print("for solver: %s, lnorms: %s, type: %s" % (sol_ver, ['l2'], multiC))
print("")
logreg = Pipeline([('classifier', LogisticRegression(solver=sol_ver, max_iter=iters, multi_class=multiC))])
grid_search = GridSearchCV(logreg, param_grid, cv=5, verbose=True)
grid_search.fit(X_train, y_train)
top_grid_score = grid_search.score(X_test, y_test)
print("best LogistRegression accuracy from grid search: %.3f" % top_grid_score)
print("did classifier get us best score already? top_grid_socre==%.4f, clf_fit_score==%.4f" % (top_grid_score, clf_fit_score))
print("##########GridSearch Analysis Done############")
print("")

# ########### WE'VE DONE GRID-SEARCH BUT LETS ALSO LOOK AT CROSS-VALIDATION
cvs = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
# ad-hoc look at the difference here:
cvs.mean()
print("naive look at accruacy loss between clf and cross_val: %.5f" % abs((cvs.mean() - clf_fit_score)))
y_train_predictions = cross_val_predict(clf, X_train, y_train, cv=5)
mse = mean_squared_error(y_train, y_train_predictions, multioutput='raw_values')  # need to interpret this!

# ################# PRECISION AND RECALL REQUIRE LESS INTERPRETATION
# compare actual labels with predicted lablels e.g(1,2,3,4,5)
# these are multiclass so we'll get an array for each class (row) while each column is the number of
# predicted instances. position in each "array" is the class.
cf = confusion_matrix(y_train, y_train_predictions)
cf_perfect = confusion_matrix(y_train, y_train)

# this is easier to see if we make dataframe for the matrices
# we want to see high scores along the diagonal
df_cf = pd.DataFrame(cf)
df_cf_perfect = pd.DataFrame(cf_perfect)
print("")
print("PERFECT Confusion Matrix For Categories\n", df_cf_perfect)
print("")
print("Confusion Matrix For Categories\n", df_cf)

# note on precisiona and recall
'''
    'binary':
    Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.

    'micro':
    Calculate metrics globally by counting the total true positives, false negatives and false positives.

    'macro':
    Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

    'weighted':
    Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

    'samples':
    Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
'''
prec = precision_score(y_train, y_train_predictions, average='micro')
rec = recall_score(y_train, y_train_predictions, average='micro')
print("precision %.4f and recall %.4f" % (prec, rec))

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
