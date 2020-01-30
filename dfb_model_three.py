import warnings
warnings.filterwarnings(action="ignore", message="^.*LAPACK bug 0038.*")
warnings.filterwarnings(action="ignore", message="^.*n_estimators.*")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

'''
Goal:
    Predict the Target for both the random and bias data sets
Specific:
    Find a good model... 
'''
#target_categories = [['Medicine', 1], ['GovBiz', 2], ['Education', 3], ['Arts', 4], ['Stem', 5]]

df = pd.read_csv('bias_student_metrics.csv')
# df = pd.read_csv('random_student_metrics.csv')

# ######### ALTERNATIVE WAY TO BUILD
# ##### https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
df = pd.read_csv('bias_student_metrics.csv')
X = df.drop('Target', axis=1)
y = df['Target'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
num_transform = Pipeline([('mimax_scaler', MinMaxScaler())])
cat_transform = Pipeline([('onehot', OneHotEncoder(categories='auto'))])
preprocessor = ColumnTransformer(
    [
        ('num', num_transform, numeric_features),
        ('cat', cat_transform, categorical_features)
    ]
)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])
clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10, 100],
}

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(X_train, y_train)

print(("best logistic regression from grid search: %.3f"
       % grid_search.score(X_test, y_test)))
