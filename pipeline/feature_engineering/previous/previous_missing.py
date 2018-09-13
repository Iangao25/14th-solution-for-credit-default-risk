import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def previous_missing(df, origin,feat):
    for k in feat:
        if origin[k].isnull().sum() > 0:
            miss_count = origin.loc[origin[k].isnull(),
            ['SK_ID_CURR','SK_ID_PREV']].groupby(['SK_ID_CURR']).count()
            miss_count.columns = ["prev_"+ k + '_miss_times']
            miss_count.reset_index(inplace = True)
            df = pd.merge(df, miss_count, on = ['SK_ID_CURR'], how = 'left')
    return df
