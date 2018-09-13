import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from contextlib import contextmanager
from Basic_function import reduce_mem_usage
from Basic_function import import_data
from Basic_function import correlation_reduce
from Basic_function import timer
from creditcard.card_missing import card_missing
from creditcard.card_sk import card_sk
from creditcard.card_using import card_using
from creditcard.card_all import card_all
from creditcard.card_amt_total_payment import card_amt_total_payment
from creditcard.card_last_one_year import card_last_one_year
from creditcard.card_last_two_year import card_last_two_year
import warnings
warnings.filterwarnings("ignore")

def card_feature(df,Debug = False):
    if Debug:
        credit = import_data("D:\Kaggle\MyFirstKaggleCompetition\Data\credit_card_balance.csv")
        credit = credit.sample(10000)
    else:
        credit = import_data("D:\Kaggle\MyFirstKaggleCompetition\Data\credit_card_balance.csv")
    card_main = df[['SK_ID_CURR']]
    key = ['SK_ID_CURR','SK_ID_PREV']
    amt = [f for f in credit.columns if 'AMT' in f]
    cnt = [f for f in credit.columns if 'CNT' in f]
    sk = ['SK_DPD', 'SK_DPD_DEF']
    with timer("card missing analysis"):
        card_main = card_missing(card_main,credit)
    with timer("card overdue analysis"):
        card_main = card_sk(card_main, credit, key + sk)
    with timer("card using analysis"):
        card_main = card_using(card_main, credit, key + amt)
    with timer("card all behavior analysis"):
        card_main = card_all(card_main, credit, key + amt + cnt)
    with timer("card using behavior analysis from the first payment of AMT_PAYMENT_TOTAL_CURRENT"):
        card_main = card_amt_total_payment(card_main, credit, key + amt + cnt)
    with timer("card last two year behavior analysis"):
        card_main = card_last_two_year(card_main, credit)
    with timer("card last one year behavior analysis"):
        card_main = card_last_one_year(card_main, credit)
    card_main.fillna(0,inplace = True)
    card_main = correlation_reduce(card_main)
    df = pd.merge(df, card_main, on = ['SK_ID_CURR'], how = 'left')
    return df
