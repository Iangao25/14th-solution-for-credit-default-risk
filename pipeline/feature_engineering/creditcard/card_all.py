import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def card_all(df, origin,feat):
    temp = origin[feat]
    temp['AMT_RECEIVABLE_DIFF'] = temp.AMT_RECIVABLE - temp.AMT_RECEIVABLE_PRINCIPAL
    temp['AMT_BALANCE_DIFF'] = temp.AMT_CREDIT_LIMIT_ACTUAL - temp.AMT_BALANCE
    temp['AMT_DRAWINGS_DIFF'] = temp.AMT_CREDIT_LIMIT_ACTUAL - temp.AMT_DRAWINGS_CURRENT
    #ratio variable
    temp['AMT_CREDIT_RATIO'] = temp.AMT_DRAWINGS_CURRENT/(1 + temp.AMT_CREDIT_LIMIT_ACTUAL)
    temp['AMT_BALANCE_RATIO'] = temp.AMT_BALANCE/(1 + temp.AMT_CREDIT_LIMIT_ACTUAL)
    temp['AMT_RECEIVABLE_RATIO'] = temp.AMT_RECIVABLE/(1 + temp.AMT_BALANCE)
    temp = temp.groupby(['SK_ID_CURR','SK_ID_PREV']).agg({
    k:['mean','min','max','median','sum','std'] for k in temp.columns
    if k not in ['SK_ID_CURR','SK_ID_PREV']})
    temp = correlation_reduce(temp)
    temp.columns = ["_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    temp = temp.groupby(['SK_ID_CURR']).agg({
    k:['mean','min','max','median','sum','std'] for k in temp.columns
    if k not in ['SK_ID_CURR','SK_ID_PREV']})
    temp.columns = ["card_all_" + "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    temp = correlation_reduce(temp)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
