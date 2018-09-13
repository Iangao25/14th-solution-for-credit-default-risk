import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def pos_cash_last(df, origin):
    temp = origin.groupby(['SK_ID_CURR','MONTHS_BALANCE']).agg(
    {'CNT_INSTALMENT_FUTURE':['mean','min','max','sum']}).reset_index().sort_values(
    ['SK_ID_CURR','MONTHS_BALANCE'],ascending = False)
    temp = temp.groupby('SK_ID_CURR').first()
    temp.columns = ["pos_last_" + "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
