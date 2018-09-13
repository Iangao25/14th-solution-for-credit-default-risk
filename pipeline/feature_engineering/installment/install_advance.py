import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce
from scipy.stats import kurtosis, iqr, skew

def install_advance(df, origin):
    temp1 = origin.groupby(['SK_ID_CURR']).agg(
    {k:['sum','mean','median','min','max','std',iqr,skew,kurtosis]
    for k in ['instalment_paid_late_in_days','instalment_paid_over_amount']})
    temp2 = origin.groupby(['SK_ID_CURR']).agg(
    {k:['sum','mean','count'] for k in ['instalment_paid_late','instalment_paid_over']}
    )
    temp1.columns = ["_".join(j) for j in temp1.columns.ravel()]
    temp2.columns = ["_".join(j) for j in temp2.columns.ravel()]
    temp2 = correlation_reduce(temp2)
    temp1.reset_index(inplace = True)
    temp2.reset_index(inplace = True)
    df = pd.merge(df, temp1, on = ['SK_ID_CURR'], how = 'left')
    df = pd.merge(df, temp2, on = ['SK_ID_CURR'], how = 'left')
    return df
