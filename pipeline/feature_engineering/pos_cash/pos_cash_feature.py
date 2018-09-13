import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from contextlib import contextmanager
from Basic_function import reduce_mem_usage
from Basic_function import import_data
from Basic_function import correlation_reduce
from Basic_function import timer
from pos_cash.pos_cash_basic import pos_cash_basic
from pos_cash.pos_cash_last import pos_cash_last
from pos_cash.pos_cash_last_k_installment import pos_cash_last_k_installment
from pos_cash.pos_cash_last_loan import pos_cash_last_loan
from pos_cash.pos_cash_trend_installment import pos_cash_trend_installment
def pos_cash_feature(df,Debug = False):
    if Debug:
        pos = import_data("D:\Kaggle\MyFirstKaggleCompetition\Data\POS_CASH_balance.csv")
        pos = pos.sample(10000)
    else:
        pos = import_data("D:\Kaggle\MyFirstKaggleCompetition\Data\POS_CASH_balance.csv")
    pos_main = df[['SK_ID_CURR']]
    pos['SK_DPD_DIFF'] = pos.SK_DPD - pos.SK_DPD_DEF
    pos['pos_cash_paid_late'] = pos['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    pos['pos_cash_paid_late_with_tolerance'] = pos['SK_DPD_DEF'].apply(lambda x: 1 if x > 0 else 0)
    with timer("pos basic stat analysis"):
        pos_main = pos_cash_basic(pos_main, pos)
    with timer("pos cash last record"):
        pos_main = pos_cash_last(pos_main, pos)
    with timer("pos cash last k installment analysis"):
        pos_main = pos_cash_last_k_installment(pos_main, pos)
    with timer("pos cash last loan analysis"):
        pos_main = pos_cash_last_loan(pos_main, pos)
    with timer("pos cash trend analysis"):
        pos_main = pos_cash_trend_installment(pos_main, pos)
    pos_main.fillna(0, inplace = True)
    df = pd.merge(df, pos_main, on = ['SK_ID_CURR'], how = 'left',validate='one_to_one')
    return df
