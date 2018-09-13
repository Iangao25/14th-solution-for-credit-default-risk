import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import correlation_reduce

def previous_category(df, origin, feat):
    for i in feat:
        temp = origin[['SK_ID_PREV','SK_ID_CURR',i]].groupby(
        ['SK_ID_CURR',i],as_index=True).count().unstack().reset_index()
        temp.columns = ['SK_ID_CURR'] + ["prev" + "_" +'_'.join((temp.columns.names[1],
        str(col[1]))) for col in temp.columns[1:]]
        df = pd.merge(df, temp, how = 'left' ,on = ['SK_ID_CURR'])
        print("feature" + "_" + i + " is over.")
    return df
