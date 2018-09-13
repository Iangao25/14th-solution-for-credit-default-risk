import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def install_basic(df,origin):
    temp = origin.groupby(['SK_ID_CURR']).agg(
    {k:['mean','min','max','median','sum','std'] for k in
     ['AMT_INSTALMENT','AMT_PAYMENT','DAYS_ENTRY_PAYMENT',
     'DAYS_INSTALMENT','NUM_INSTALMENT_NUMBER','NUM_INSTALMENT_VERSION']})
    temp.columns = ["ins_" + "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
