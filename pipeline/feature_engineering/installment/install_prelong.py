import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def install_prelong(df, origin):
    temp = origin.sort_values(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER']).drop_duplicates(
    ['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_VERSION'])
    temp['pay_long'] = temp[['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER']].groupby(['SK_ID_CURR','SK_ID_PREV']).shift(-1)
    temp.pay_long = temp.pay_long - temp.NUM_INSTALMENT_NUMBER
    temp.pay_long.fillna(1, inplace = True)
    temp['pay_long_amount'] = temp.AMT_PAYMENT * temp.pay_long
    temp1 = temp.astype(np.float64).groupby(['SK_ID_CURR']).agg(
    {k:['mean','sum','min','median','max','std'] for k in ['pay_long','pay_long_amount']})
    temp1.columns = ["ins" + "_" + col1 + "_" + col2 for col1 in temp1.columns.levels[0] for col2 in temp1.columns.levels[1]]
    temp1.reset_index(inplace = True)
    temp1 = correlation_reduce(temp1)
    df = pd.merge(df, temp1, on = ['SK_ID_CURR'], how = 'left')
    return df
