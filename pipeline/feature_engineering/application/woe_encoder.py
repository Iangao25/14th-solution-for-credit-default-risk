import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from Basic_function import reduce_mem_usage
from Basic_function import import_data

def woe_encoder(df):
    print("woe categorical analysis.")
    from math import log
    train = import_data("D:\\Kaggle\\MyFirstKaggleCompetition\\Data\\application_train.csv")
    categorical_feats = [f for f in train.columns if train[f].dtype == 'object']
    temp = train[['SK_ID_CURR','TARGET'] + categorical_feats]
    woe_main = df[['SK_ID_CURR'] + categorical_feats]
    for i in categorical_feats:
        temp1 = temp[['SK_ID_CURR',i,'TARGET']].groupby([i,'TARGET']).count().unstack()
        temp1.columns = [temp1.columns.names[1] + "_" + str(col) for col in temp1.columns.levels[1]]
        temp1.loc['Row_sum'] = temp1.apply(lambda x: x.sum())
        temp1['WOE'] = map(lambda x,y:
        log((float(x)/temp1.loc['Row_sum','TARGET_1'])/(float(y)/temp1.loc['Row_sum','TARGET_0'])),
        temp1.TARGET_1,temp1.TARGET_0)
        temp1.drop('Row_sum',axis = 0,inplace = True)
        temp1.drop(['TARGET_0','TARGET_1'], axis = 1,inplace = True)
        temp1.columns = [temp1.index.name + "_woe"]
        temp1.reset_index(inplace = True)
        woe_main = pd.merge(woe_main, temp1, on = [i], how = 'left')
        print("feature" + "_" + i + " is finished!")
    woe_main.fillna(0,inplace  = True)
    woe_main.drop(categorical_feats, axis = 1,inplace = True)
    df = pd.merge(df,woe_main, on = ['SK_ID_CURR'], how = 'left')
    return df
