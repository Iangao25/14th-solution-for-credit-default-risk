import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import reduce_mem_usage
from Basic_function import correlation_reduce

def bureau_overdue(df,origin):
    temp = origin.groupby(['SK_ID_CURR','SK_ID_BUREAU']).agg({"overdue":['mean','sum']})
    temp.columns = ["_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    temp = temp.astype(np.float64).groupby(['SK_ID_CURR']).agg(
    {k:['mean','max','std','sum','median'] for k in temp.columns if k not in ['SK_ID_CURR','SK_ID_BUREAU']})
    temp.columns = ['bureau_' + "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    temp = correlation_reduce(temp)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
