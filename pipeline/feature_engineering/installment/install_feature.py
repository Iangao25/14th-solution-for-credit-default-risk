import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from contextlib import contextmanager
from Basic_function import reduce_mem_usage
from Basic_function import import_data
from Basic_function import correlation_reduce
from Basic_function import timer
from Basic_function import parallel_apply
from installment.install_basic import install_basic
from installment.install_advance import install_advance
from installment.install_prelong import install_prelong
from installment.install_last_k_feature import install_last_k_feature
from installment.install_trend_k_feature import install_trend_k_feature
from installment.install_last_k_fraction_feature import install_last_k_fraction_feature

import warnings
warnings.filterwarnings("ignore")

def install_feature(df,Debug = False):
    if Debug:
        installment = import_data("D:\Kaggle\MyFirstKaggleCompetition\Data\installments_payments.csv")
        installment = installment.sample(10000)
    else:
        installment = import_data("D:\Kaggle\MyFirstKaggleCompetition\Data\installments_payments.csv")
    ins_main = df[['SK_ID_CURR']]
    installment['instalment_paid_late_in_days'] = installment['DAYS_ENTRY_PAYMENT'] - installment['DAYS_INSTALMENT']
    installment['instalment_paid_late'] = installment['instalment_paid_late_in_days'].apply(lambda x: 1 if x > 0 else 0)
    installment['instalment_paid_over_amount'] = installment['AMT_PAYMENT'] - installment['AMT_INSTALMENT']
    installment['instalment_paid_over'] = installment['instalment_paid_over_amount'].apply(lambda x: 1 if x > 0 else 0)
    with timer("basic stat analysis"):
        ins_main = install_basic(ins_main, installment)
    with timer("advance stat analysis"):
        ins_main = install_advance(ins_main, installment)
    with timer("install prelong analysis"):
        ins_main = install_prelong(ins_main, installment)
    with timer("last k installment analysis"):
        ins_main = install_last_k_feature(ins_main, installment)
    with timer("last k fraction installment analysis"):
        ins_main = install_last_k_fraction_feature(ins_main, installment)
    with timer("last k trend installment analysis"):
        ins_main = install_trend_k_feature(ins_main, installment)
    ins_main.fillna(0, inplace = True)
    df = pd.merge(df, ins_main, how = 'left', on = ['SK_ID_CURR'],validate='one_to_one')
    return df
