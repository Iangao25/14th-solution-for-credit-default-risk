import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from Basic_function import reduce_mem_usage
from Basic_function import import_data
from Basic_function import correlation_reduce
from Basic_function import timer
from contextlib import contextmanager
from bureau_balance.bureau_balance_missing import  bureau_balance_missing
from bureau_balance.bureau_overdue import bureau_overdue
from bureau_balance.bureau_status import bureau_status
import warnings
warnings.filterwarnings("ignore")

def bureau_balance_feature(df,Debug = False):
    if Debug:
        bureau_balance = import_data("D://Kaggle//MyFirstKaggleCompetition//Data//bureau_balance.csv")
        bureau_balance = bureau_balance.sample(10000)
    else:
        bureau_balance = import_data("D://Kaggle//MyFirstKaggleCompetition//Data//bureau_balance.csv")
    bureau = import_data("D://Kaggle//MyFirstKaggleCompetition//Data//bureau.csv")
    bureau_all = pd.merge(bureau[['SK_ID_CURR','SK_ID_BUREAU']],
                        bureau_balance.groupby(
                        ['SK_ID_BUREAU','STATUS'], as_index = False).count(),how = 'left', on = ['SK_ID_BUREAU'])
    bureau_main = df[['SK_ID_CURR']]
    bureau_all['overdue'] = bureau_all.STATUS.apply(lambda x: 1 if x in ['1','2','3','4','5'] else 0)
    with timer("bureau balance missing count"):
        bureau_main = bureau_balance_missing(bureau_main,bureau_all)
    with timer("bureau balance overdue analysis"):
        bureau_main = bureau_overdue(bureau_main, bureau_all)
    with timer("bureau balance status count"):
        bureau_main = bureau_status(bureau_main, bureau_all)
    bureau_main.fillna(0, inplace = True)
    bureau_main = correlation_reduce(bureau_main)
    df = pd.merge(df, bureau_main, on = ['SK_ID_CURR'], how = 'left',validate='one_to_one')
    return df
