import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def bureau_flag(df,origin):
    temp = origin.groupby(['SK_ID_CURR']).agg(
    {k:['mean','sum','count'] for k in origin.columns if 'flag' in k})
    temp.columns = ["credit_" + "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
