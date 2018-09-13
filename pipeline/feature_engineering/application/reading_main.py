
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from Basic_function import reduce_mem_usage
from Basic_function import import_data
def reading_main():
    target_label = 'TARGET'
    train = import_data("D:\\Kaggle\\MyFirstKaggleCompetition\\Data\\application_train.csv")
    df_x_submission = import_data("D:\\Kaggle\\MyFirstKaggleCompetition\\Data\\application_test.csv")
    np.random.seed(1222)
    train_set_size = int(round(train.shape[0] * 0.9))
    df_x_train, df_x_test     = np.split(train.sample(frac = 1), [train_set_size])
    y_train = df_x_train[['SK_ID_CURR',target_label]]
    y_test = df_x_test[['SK_ID_CURR',target_label]]
    df_x_train = df_x_train.drop(target_label, axis = 1)
    df_x_test = df_x_test.drop(target_label, axis = 1)
    df_x_train['is_train'] = 1
    df_x_test['is_train'] = 0
    df_x_submission['is_train'] = -1
    # Concatenate everything
    main = pd.concat([df_x_train, df_x_test, df_x_submission])
    return main, y_train, y_test
