import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def pos_cash_basic(df, origin):
    temp = origin.groupby(['SK_ID_CURR']).agg(
    {k:['mean','median','min','max','sum','std']
    for k in ['SK_DPD','SK_DPD_DEF','MONTHS_BALANCE','SK_DPD_DIFF']})
    temp.columns = ["pos_" + "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    temp = correlation_reduce(temp)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
