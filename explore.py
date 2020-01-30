import pandas as pd
import numpy as np
from collections import Counter


'''
Goal:
    Predict the Target for both the random and bias data sets
Specific:
    Explore the dataset...
'''

dfr = pd.read_csv('random_student_metrics.csv')
dfb = pd.read_csv('bias_student_metrics.csv')

cr = Counter(dfr.Target)
cb = Counter(dfb.Target)

# ########### MAKE DATA SIMILAR
general_attr = ['Social', 'Spatial', 'Temporal', 'Organizational', 'Verbal']
specific_attr = ['TechnicalWriting', 'DescriptiveWriting', 'AnalyticWriting', 'Arithmetic', 'Algebra', 'Geometry']
dfr[general_attr] = dfr[general_attr].astype(float)
dfr[['GenderLabel', 'Target']] = dfr[['GenderLabel', 'Target']].astype('category')
dfb[general_attr] = dfb[general_attr].astype(float)
dfb[['GenderLabel', 'Target']] = dfb[['GenderLabel', 'Target']].astype('category')


# ######## LOOK AT GENERAL ATTRIBUTES
import matplotlib.pyplot as plt


def fig_df_bins(df, figname):
    """
        fig_df_bins(dfr, 'rand_histo_attributes.png')
        fig_df_bins(dfb, 'bias_histo_attributes.png')
    """
    df.hist(bins=50, figsize=(15, 10))
    plt.savefig(figname)


fig_df_bins(dfr, 'rand_histo_attributes.png')
fig_df_bins(dfb, 'bias_histo_attributes.png')


# ######## LOOK AT SAMPLING
def make_stratum(df, strata):
    aa = np.partition(dfr.Age, strata+1)
    l = [v for v in range(strata+1)]
    idxs = [round(len(df) * x/(strata+1)) for x in l]
    b = [aa[x] for x in idxs]
    b[-1] = np.inf
    return b, l


def stratify_age(df, name):
    '''
    verify the age sampling is properly stratified
    THOUGH, we already know how it was "sampled" since we generated the data ;)
    # b, l = make_stratum(df, strata)
    # df['age_cat'] = pd.cut(df['Age'], bins=b, labels=l[1:])
    '''
    df['age_cat'] = pd.cut(df['Age'], bins=[0, 10, 16, 21, 25, np.inf], labels=[1, 2, 3, 4, 5])
    df.age_cat.hist()
    plt.savefig(name)


stratify_age(dfr, 'random_age_strat.png')
stratify_age(dfb, 'bias_age_strat.png')

from sklearn.model_selection import StratifiedShuffleSplit


def make_stratified_age(df):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df.age_cat):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return (strat_train_set, strat_test_set)


r_strat_train, r_strat_test = make_stratified_age(dfr)
b_strat_train, b_strat_test = make_stratified_age(dfb)


def age_cat_proportion(data):
    return data.age_cat.value_counts() / len(data)


from sklearn.model_selection import train_test_split
r_train_set, r_test_set = train_test_split(dfr, test_size=0.2, random_state=42)
b_train_set, b_test_set = train_test_split(dfb, test_size=0.2, random_state=42)


def compare_props(df, strat_test, test_set):
    compare_props = pd.DataFrame({
        "Overall": age_cat_proportion(df),
        "Stratified": age_cat_proportion(strat_test),
        "Random": age_cat_proportion(test_set),
    }).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"]-100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"]-100
    return compare_props


r_strat_props = compare_props(dfr, r_strat_test, r_test_set)
b_strat_props = compare_props(dfb, b_strat_test, b_test_set)

# ######## LOOK AT CORRELATIONS (linear assumption => is there some kind of linear corelation we can see?)
r_corr_mat = dfr.corr()
b_corr_mat = dfb.corr()

r_corr_mat.Age.sort_values(ascending=False)
b_corr_mat.Age.sort_values(ascending=False)

from pandas.plotting import scatter_matrix

scatter_matrix(dfr[general_attr], figsize=(12, 8))
scatter_matrix(dfb[general_attr], figsize=(12, 8))

scatter_matrix(dfr[specific_attr], figsize=(12, 8))
scatter_matrix(dfb[specific_attr], figsize=(12, 8))
# ############################## these are all pretty bad... nothing really sticks out

# correlations were flat, this means we shoudl probably combine some features or do some dimension reduction
dfb['general_sum'] = dfb[general_attr].sum(axis=1)
dfb['specific_sum'] = dfb[specific_attr].sum(axis=1)

dfr['general_sum'] = dfr[general_attr].sum(axis=1)
dfr['specific_sum'] = dfr[specific_attr].sum(axis=1)

r_corr_mat = dfr.corr()
b_corr_mat = dfb.corr()

scatter_matrix(dfr[['Target', 'general_sum', 'specific_sum']], figsize=(12, 8))
scatter_matrix(dfb[['Target', 'general_sum', 'specific_sum']], figsize=(12, 8))
# ##################### BUSTED ... still nothing really standing out...
