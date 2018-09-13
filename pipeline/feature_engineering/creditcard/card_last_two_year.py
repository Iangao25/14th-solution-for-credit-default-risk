import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def card_last_two_year(df, origin):
    temp = origin.loc[(origin.MONTHS_BALANCE >= -24) & (origin.NAME_CONTRACT_STATUS == 'Active'),
    ['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_CURRENT',
    'AMT_PAYMENT_CURRENT','AMT_RECEIVABLE_PRINCIPAL',
    'CNT_DRAWINGS_CURRENT','SK_DPD','SK_DPD_DEF']].sort_values(['SK_ID_CURR','MONTHS_BALANCE'])
    temp['amt_balance_ratio'] = temp.AMT_BALANCE/(temp.AMT_CREDIT_LIMIT_ACTUAL + 1)
    temp1 = temp.astype(np.float64).groupby(['SK_ID_CURR','SK_ID_PREV']).agg(
    {k:['min','median','max','mean','std','sum']
    for k in temp.columns if k not in ['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE']})
    temp1.columns = ["_".join(j) for j in temp1.columns.ravel()]
    temp1.reset_index(inplace = True)
    temp1 = temp1.groupby(['SK_ID_CURR']).agg({
    k:['min','max','mean','median','std','sum']
    for k in temp1.columns if k not in ['SK_ID_CURR','SK_ID_PREV']})
    temp1.columns = ["card_2_year_" + "_".join(j) for j in temp1.columns.ravel()]
    temp1 = correlation_reduce(temp1)
    temp1.reset_index(inplace = True)
    df = pd.merge(df, temp1, on = ['SK_ID_CURR'], how = 'left')
    return df
