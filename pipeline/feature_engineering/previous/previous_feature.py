import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from contextlib import contextmanager
from Basic_function import reduce_mem_usage
from Basic_function import import_data
from Basic_function import correlation_reduce
from Basic_function import timer
from previous.previous_missing import previous_missing
from previous.previous_all_stat import previous_all_stat_amt
from previous.previous_all_stat import previous_all_stat_day
from previous.previous_approved import previous_approved_amt
from previous.previous_approved import previous_approved_day
from previous.previous_refused import previous_refused_amt
from previous.previous_refused import previous_refused_day
from previous.previous_category import previous_category
from previous.previous_last_k_contract import previous_last_k_contract
import warnings
warnings.filterwarnings("ignore")

def previous_feature(df,Debug = False):
    if Debug:
        prev = import_data("D://Kaggle//MyFirstKaggleCompetition//Data//previous_application.csv")
        prev = prev.sample(10000)
    else:
        prev = import_data("D://Kaggle//MyFirstKaggleCompetition//Data//previous_application.csv")
    prev_main = df[['SK_ID_CURR']]
    key = ['SK_ID_CURR','SK_ID_PREV']
    Behaviour_variable = ['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_DOWN_PAYMENT','AMT_GOODS_PRICE','RATE_DOWN_PAYMENT','CNT_PAYMENT']
    Days_variable = ['DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION','DAYS_DECISION']
    Categorical_variable = ['NAME_CONTRACT_TYPE','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START',
                        'NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE','CODE_REJECT_REASON','NAME_TYPE_SUITE',
                       'NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY',
                       'NAME_YIELD_GROUP','PRODUCT_COMBINATION']
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    with timer("previous application missiong count analysis."):
        prev_main = previous_missing(prev_main, prev, Behaviour_variable + Days_variable)
    with timer("previous all record analysis for amt variable."):
        prev_main = previous_all_stat_amt(prev_main, prev, key + Behaviour_variable)
    with timer("previous all record analysis for day variable."):
        prev_main = previous_all_stat_day(prev_main, prev, key + Days_variable)
    with timer("previous approved analysis for amt variable"):
        prev_main = previous_approved_amt(prev_main, prev, key + Behaviour_variable)
    with timer("previous approved analysis for day variable"):
        prev_main = previous_approved_day(prev_main, prev, key + Days_variable)
    with timer("previous refused analysis for amt variable"):
        prev_main = previous_refused_amt(prev_main, prev, key + Behaviour_variable)
    with timer("previous refused analysis for day variable"):
        prev_main = previous_refused_day(prev_main, prev, key + Days_variable)
    with timer("previous category variable analysis."):
        prev_main = previous_category(prev_main, prev, Categorical_variable)
    with timer("previous last k contract analysis."):
        prev_main = previous_last_k_contract(prev_main, prev)
    prev_main.fillna(0, inplace = True)
    prev_main = correlation_reduce(prev_main)
    df = pd.merge(df, prev_main, on = ['SK_ID_CURR'], how = 'left',validate='one_to_one')
    df = reduce_mem_usage(df)
    return df
