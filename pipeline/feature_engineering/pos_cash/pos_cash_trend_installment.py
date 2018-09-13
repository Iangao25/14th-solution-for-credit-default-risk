import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce
from scipy.stats import kurtosis, iqr, skew
from Basic_function import parallel_apply
from functools import partial
from pos_cash.pos_cash_function import trend_in_last_k_installment_features

def pos_cash_trend_installment(df, origin):
    groupby = origin.groupby(['SK_ID_CURR'])
    func = partial(trend_in_last_k_installment_features, periods=[1,6,12,30,60])
    g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=8).reset_index()
    df = pd.merge(df, g, on = ['SK_ID_CURR'], how = 'left')
    return df
