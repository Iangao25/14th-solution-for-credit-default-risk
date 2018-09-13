import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def card_sk(df,origin,feat):
    temp = origin[feat].astype(np.float64)
    temp['SK_out'] = temp.SK_DPD - temp.SK_DPD_DEF
    # temp['SK_ratio'] = temp.SK_
    temp['card_out_overdue_flag'] = temp.SK_out.apply(lambda x: 1 if x > 0 else 0)
    # temp['card_out_overdue_flag'] = map(lambda x: 1 if x > 0 else 0, temp.SK_out)
    print("flag variable stat")
    temp1 = temp.groupby('SK_ID_CURR').agg(
    {k:['mean','sum','count'] for k in ['card_out_overdue_flag']})
    temp1.columns = ["card_" + "_".join(j) for j in temp1.columns.ravel()]
    temp1.reset_index(inplace = True)
    temp1 = correlation_reduce(temp1)
    df = pd.merge(df, temp1, on = ['SK_ID_CURR'], how = 'left')
    print("days continous variable stat")
    temp2 = temp.groupby('SK_ID_CURR').agg({k:['median','mean','sum','max','std'] for k in ['SK_out','SK_DPD_DEF','SK_DPD']})
    temp2.columns = ["_".join(j) for j in temp2.columns.ravel()]
    temp2.reset_index(inplace = True)
    temp2 = correlation_reduce(temp2)
    df = pd.merge(df, temp2, on = ['SK_ID_CURR'], how = 'left')
    return df
