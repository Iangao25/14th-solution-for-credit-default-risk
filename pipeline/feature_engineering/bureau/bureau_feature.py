import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import warnings
warnings.filterwarnings("ignore")
from contextlib import contextmanager
from Basic_function import reduce_mem_usage
from Basic_function import import_data
from Basic_function import correlation_reduce
from Basic_function import timer
from bureau.bureau_missing import bureau_missing
from bureau.bureau_flag import bureau_flag
from bureau.bureau_all_debt import bureau_all_debt
from bureau.bureau_active import bureau_active
from bureau.bureau_closed import bureau_closed
def bureau_feature(df,Debug = False):
    if Debug:
        bureau = import_data("D://Kaggle//MyFirstKaggleCompetition//Data//bureau.csv")
        bureau = bureau.sample(10000)
    else:
        bureau = import_data("D://Kaggle//MyFirstKaggleCompetition//Data//bureau.csv")

    bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan
    bureau['active_flag'] = bureau.CREDIT_ACTIVE.apply(lambda x: 1 if x == 'Active' else 0)
    bureau['enddate_flag'] = bureau.DAYS_CREDIT_ENDDATE.apply(lambda x: 1 if x > 0 else 0)
    bureau['overdue_flag'] = bureau.AMT_CREDIT_MAX_OVERDUE.apply(lambda x: 1 if x > 0 else 0)
    bureau['using_flag'] = bureau.AMT_CREDIT_SUM_DEBT.apply(lambda x: 1 if x > 0 else 0)
    credit_main = pd.DataFrame({'SK_ID_CURR': df['SK_ID_CURR'].unique()})
    day = ['DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT','DAYS_CREDIT','DAYS_CREDIT_UPDATE']
    key = ['SK_ID_CURR','SK_ID_BUREAU']
    amt = ['CNT_CREDIT_PROLONG','CREDIT_DAY_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT',
           'AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE','AMT_ANNUITY']
    cat = ['CREDIT_ACTIVE','CREDIT_CURRENCY','CREDIT_TYPE']
    with timer("bureau missing count"):
        credit_main = bureau_missing(credit_main, bureau)
        print(credit_main.shape)
    with timer("bureau flag variable analysis"):
        credit_main = bureau_flag(credit_main, bureau)
        print(credit_main.shape)
    with timer("bureau amt debt analysis"):
        credit_main = bureau_all_debt(credit_main, bureau)
        print(credit_main.shape)
    with timer("bureau active status analysis"):
        credit_main = bureau_active(credit_main, bureau, key + day + amt)
        print(credit_main.shape)
    with timer("bureau closed status analysis"):
        credit_main = bureau_closed(credit_main, bureau, key + day + amt)
        print(credit_main.shape)

    credit_main.fillna(0, inplace = True)
    df = pd.merge(df, credit_main, on = ['SK_ID_CURR'],how = 'left',validate='one_to_one')
    return df
