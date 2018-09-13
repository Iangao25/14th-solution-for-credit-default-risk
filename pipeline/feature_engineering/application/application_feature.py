import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from Basic_function import reduce_mem_usage
from Basic_function import correlation_reduce

def application_feature(df):
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    df['annuity_income_percentage'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['car_to_birth_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['car_to_employ_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['children_ratio'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['credit_to_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['credit_to_goods_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['credit_to_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['days_employed_percentage'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['income_credit_percentage'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['income_per_child'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['income_per_person'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['payment_rate'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['phone_to_birth_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['phone_to_employ_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['edfternal_sources_weighted'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4
    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
        df['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
        df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    df['short_employment'] = df['DAYS_EMPLOYED'].apply(lambda x: 1 if x > -2000 else 0)
    df['young_age'] = df['DAYS_BIRTH'].apply(lambda x: 1 if x > -14000 else 0)
    return df
