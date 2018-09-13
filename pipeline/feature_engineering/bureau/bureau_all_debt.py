import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def bureau_all_debt(df, origin):
    temp = origin.groupby(['SK_ID_CURR']).agg(
    {'AMT_CREDIT_SUM_DEBT':'sum','AMT_CREDIT_SUM':'sum','AMT_CREDIT_SUM_OVERDUE':'sum'})
    temp.columns = ['bureau_total_customer_debt',
                    'bureau_total_customer_credit',
                    'bureau_total_customer_overdue']
    temp['bureau_debt_credit_ratio'] = \
    temp['bureau_total_customer_debt'] / (1 + temp['bureau_total_customer_credit'])
    temp['bureau_overdue_debt_ratio'] = \
    temp['bureau_total_customer_overdue'] / (1 + temp['bureau_total_customer_debt'])
    temp.reset_index(inplace = True)
    df = pd.merge(df, temp, on = ['SK_ID_CURR'], how = 'left')
    return df
