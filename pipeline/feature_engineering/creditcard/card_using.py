import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def card_using(df, origin,feat):
    temp = origin[feat]
    temp['AMT_BALANCE_delay'] = temp.groupby(['SK_ID_CURR','SK_ID_PREV']).AMT_BALANCE.shift()
    temp.AMT_BALANCE_delay.fillna(temp.AMT_BALANCE, inplace = True)
    temp['AMT_pay'] = temp.AMT_BALANCE_delay - temp.AMT_BALANCE
    temp['card_using_flag'] = temp.AMT_BALANCE.apply(lambda x: 1 if x > 0 else 0)
    # temp['card_using_flag'] = map(lambda x: 1 if x > 0 else 0,temp.AMT_BALANCE)
    temp1 = temp.groupby(['SK_ID_CURR']).agg({'card_using_flag':['mean','sum','count']})
    temp1.columns = ["_".join(j) for j in temp1.columns.ravel()]
    temp1.reset_index(inplace = True)
    temp1 = correlation_reduce(temp1)
    print("card using behavior analysis from the first payment of AMT_PAYMENT_TOTAL_CURRENT")
    #could do more stats
    temp2 = temp.loc[temp.AMT_PAYMENT_TOTAL_CURRENT != 0].drop_duplicates(['SK_ID_CURR','SK_ID_PREV','AMT_PAYMENT_TOTAL_CURRENT'],keep = 'last')
    temp2 = temp2.astype(np.float64).groupby(['SK_ID_CURR','SK_ID_PREV']).agg(
    {'AMT_PAYMENT_TOTAL_CURRENT':['max','min','median','mean','sum','std']})
    temp2.columns = ["_".join(j) for j in temp2.columns.ravel()]
    temp2.reset_index(inplace = True)
    temp2 = temp2.groupby(['SK_ID_CURR']).agg({
    k:['mean','min','max','sum','std','median'] for k in temp2.columns
    if k not in ['SK_ID_CURR','SK_ID_PREV']
    })
    temp2.columns = ["card_first_time_" + "_".join(j) for j in temp2.columns.ravel()]
    temp2.reset_index(inplace = True)
    temp2 = correlation_reduce(temp2)
    df = pd.merge(df, temp1, on = ['SK_ID_CURR'], how = 'left')
    df = pd.merge(df, temp2, on = ['SK_ID_CURR'], how = 'left')
    return df
