import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def previous_approved_amt(df, origin, feat):
    temp = origin.loc[origin.NAME_CONTRACT_STATUS == 'Approved', feat].astype(np.float64)
    temp['amt_min_year'] = temp.AMT_CREDIT/(1 + temp.AMT_ANNUITY)
    temp['amt_credit_trust'] = temp.AMT_CREDIT - temp.AMT_APPLICATION
    temp['amt_credit_ratio'] = temp.AMT_CREDIT/(1 + temp.AMT_APPLICATION)
    temp['amt_goods_ratio'] = temp.AMT_CREDIT/(1 + temp.AMT_GOODS_PRICE)
    temp['amt_goods_remain'] = temp.AMT_CREDIT - temp.AMT_GOODS_PRICE
    temp = temp.groupby(['SK_ID_CURR']).agg({
    k:['min','median','max','mean','std'] for k in temp.columns if k not in ['SK_ID_CURR','SK_ID_PREV']})
    temp.columns = ['prev_approved_' + "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    temp = correlation_reduce(temp)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df

def previous_approved_day(df, origin, feat):
    temp = origin.loc[origin.NAME_CONTRACT_STATUS == 'Approved',feat].astype(np.float64)
    temp['days_due_diff'] = temp.DAYS_LAST_DUE - temp.DAYS_FIRST_DUE
    temp['days_last_due_diff'] = temp.DAYS_LAST_DUE_1ST_VERSION - temp.DAYS_LAST_DUE
    temp['days_waiting_diff'] = temp.DAYS_FIRST_DUE - temp.DAYS_DECISION
    temp['days_origin_due_diff'] = temp.DAYS_LAST_DUE_1ST_VERSION - temp.DAYS_FIRST_DUE
    temp['days_origin_que_diff'] = temp.DAYS_TERMINATION - temp.DAYS_LAST_DUE
    temp = temp.groupby(['SK_ID_CURR']).agg({
    k:['min','median','max','mean','std','sum'] for k in temp.columns if k not in ['SK_ID_CURR','SK_ID_PREV']})
    temp.columns = ['prev_approved_' + "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    temp = correlation_reduce(temp)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
