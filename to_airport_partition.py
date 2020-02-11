import warnings
warnings.filterwarnings(action="ignore", message="^.*LAPACK bug 0038.*")
import numpy as np
import random
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.special import kl_div
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

PARTITION_SIZE = 20
DELAY_WEIGHTS = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.2]


def random_partition(upper_limit, psize):
    partition = [0] * psize
    for _ in range(upper_limit):
        partition[random.randrange(psize)] += 1
    return partition


def shift_delay(ahat):
    return np.average(ahat, weights=DELAY_WEIGHTS)


def divergence(p):
    A = p[:10]
    B = p[10:]
    return kl_div(A, B)


def trip_entropy(kl, ttl):
    res = (kl/ttl) + 0.0000001
    if (res == float("inf")):
        return 0.00000001
    return res


df = pd.read_csv('to_airport_data.csv')
df['alpha_hat'] = df.apply(lambda x: random_partition(x.minutes, PARTITION_SIZE), axis=1)
df['delay_average'] = df.apply(lambda x: shift_delay(x.alpha_hat), axis=1)
df['divergences'] = df.apply(lambda x: divergence(x.alpha_hat), axis=1)
df['trip_entropy'] = df.apply(lambda x: trip_entropy(sum(x.divergences), sum(x.alpha_hat)), axis=1)
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

train_drop_target = train_set.drop('minutes', axis=1)
targets = train_set['minutes'].copy()
pipeL = ColumnTransformer(
    [
        ("std_scaler_delay_and_entropy", StandardScaler(), ["trip_entropy"]),
        ("minmax_scaler_delay_and_entropy", MinMaxScaler(), ["delay_average"]),
    ]
)
train_prepared = pipeL.fit_transform(train_drop_target)


# ########### SOME DATA FOR VALIDATION
print("TrainingSet Size:", len(train_set))
print("")
some_data = train_drop_target.iloc[:10]
some_labels = targets.iloc[:10]
some_data_fitted = pipeL.fit_transform(some_data)

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
