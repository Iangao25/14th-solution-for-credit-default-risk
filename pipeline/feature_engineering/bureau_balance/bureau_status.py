import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def bureau_status(df,origin):
    temp = origin[['SK_ID_CURR','SK_ID_BUREAU','STATUS']].groupby(['SK_ID_CURR','STATUS']).count().unstack()
    temp.columns = ["balance" + "_" + temp.columns.names[1] + "_" + col for col in temp.columns.levels[1]]
    temp.fillna(0, inplace = True)
    temp['status_total'] = temp.apply(lambda x: x.sum(), axis = 1)
    for var in [k for k in temp.columns if k not in ['status_total']]:
        temp[var + "_" + "rate"] = temp[var]/temp['status_total']
    temp.reset_index(inplace = True)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
