import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def bureau_closed(df, origin, feat):
    temp = origin.loc[origin.CREDIT_ACTIVE != 'Active', feat]
    temp.fillna(0, inplace = True)
    temp['AMT_CREDIT_DEBT_RATE'] = temp.AMT_CREDIT_SUM_DEBT/(1 + temp.AMT_CREDIT_SUM)
    temp = temp.astype(np.float64).groupby(['SK_ID_CURR']).agg(
    {k:['min','median','max','mean','sum','std']
    for k in temp.columns if k not in ['DAYS_CREDIT_UPDATE'] + ['SK_ID_CURR','SK_ID_BUREAU']})
    temp.columns = ["credit_closed_"+ "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    temp = correlation_reduce(temp)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
