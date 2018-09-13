import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import reduce_mem_usage
from Basic_function import correlation_reduce

def bureau_balance_missing(df,origin):
    miss_count = origin.loc[origin.STATUS.isnull(),['SK_ID_CURR','SK_ID_BUREAU']].groupby(['SK_ID_CURR']).count()
    miss_count.columns = ['bureau_miss_time']
    miss_count.reset_index(inplace = True)
    df = pd.merge(df, miss_count, on = ['SK_ID_CURR'], how = 'left')
    return df
