import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from Basic_function import reduce_mem_usage
from Basic_function import correlation_reduce

def previous_last_k_contract(df, previous_application):
    numbers_of_applications = [1, 3, 5]
    features = pd.DataFrame({'SK_ID_CURR': previous_application['SK_ID_CURR'].unique()})
    prev_applications_sorted = previous_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
    for number in numbers_of_applications:
        prev_applications_tail = prev_applications_sorted.groupby(by=['SK_ID_CURR']).tail(number)
        group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['CNT_PAYMENT'].mean().reset_index()
        group_object.rename(index=str, columns={
        'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)},
                        inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
        group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_DECISION'].mean().reset_index()
        group_object.rename(index=str, columns={
        'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(number)},
                        inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
        group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_FIRST_DRAWING'].mean().reset_index()
        group_object.rename(index=str, columns={
        'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(number)},
                        inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
    features = correlation_reduce(features)
    features.fillna(0, inplace = True)
    df = pd.merge(df, features, on = ['SK_ID_CURR'], how = 'left')
    return df
