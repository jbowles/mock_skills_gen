import warnings
warnings.filterwarnings(action="ignore", message="^.*LAPACK bug 0038.*")
import numpy as np
import random
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def delay():
    sub_points = range(1, 16)
    return random.choice(np.random.normal(sub_points))


df = pd.read_csv('to_airport_data.csv')
df['alpha_hat'] = df.apply(lambda x: delay(), axis=1)
df['beta_hat'] = df.apply(lambda x: delay(), axis=1)
df['A_hat'] = df.apply(lambda x: x.minutes + x.alpha_hat + x.beta_hat, axis=1)
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

train_drop_target = train_set.drop('minutes', axis=1)
targets = train_set['minutes'].copy()
scaler = MinMaxScaler()
train_prepared = scaler.fit_transform(train_drop_target)


# ########### SOME DATA FOR VALIDATION
print("TrainingSet Size:", len(train_set))
print("")
some_data = train_drop_target.iloc[:10]
some_labels = targets.iloc[:10]
some_data_fitted = scaler.fit_transform(some_data)

# ########### LINEAR REGRESSION (based on what we saw in explore expect this to be ugly!)
linreg = LinearRegression()
linreg.fit(train_prepared, targets)
print("                     Targets:", list(some_labels))
print("LinearRegression Predictions:", [int(x) for x in linreg.predict(some_data_fitted)])
print("")


linreg_predictions = linreg.predict(train_prepared)
linreg_MSE = mean_squared_error(targets, linreg_predictions)
print("     LinearRegression MSE: ", np.sqrt(linreg_MSE))
linregscores = cross_val_score(linreg, train_prepared, targets, scoring="neg_mean_squared_error", cv=10)
linreg_MSE_scores = np.sqrt(-linregscores)


def display_scores(scores):
    # print("Scores", scores)
    print("              Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print("")


print("===================cross_val_score linreg_Neg_MSE")
display_scores(linreg_MSE_scores)
