import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce
from scipy.stats import kurtosis, iqr, skew
from Basic_function import parallel_apply
from functools import partial
from installment.install_function import trend_in_last_k_instalment_features

def install_trend_k_feature(df, origin):
    func = partial(trend_in_last_k_instalment_features, periods=[10,50,100,500])
    groupby = origin.groupby(['SK_ID_CURR'])
    g = parallel_apply(groupby, func, index_name='SK_ID_CURR',num_workers=8).reset_index()
    df = pd.merge(df, g, on = ['SK_ID_CURR'], how = 'left')
    return df
