import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def bureau_missing(df,origin):
    for k in origin.columns:
        if origin[k].isnull().sum() > 0:
            miss_count = origin.loc[origin[k].isnull(),['SK_ID_CURR','SK_ID_BUREAU']].groupby(['SK_ID_CURR']).count()
            miss_count.columns = ["credit_"+ k + '_miss_times']
            miss_count.reset_index(inplace = True)
            df = pd.merge(df, miss_count, on = ['SK_ID_CURR'], how = 'left')
    return df
