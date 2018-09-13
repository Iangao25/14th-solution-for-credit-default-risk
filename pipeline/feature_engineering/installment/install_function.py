
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from sklearn.linear_model import LinearRegression
from Basic_function import parallel_apply
from scipy.stats import kurtosis, iqr, skew
import warnings
warnings.filterwarnings('ignore')

def trend_in_last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)

    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]


        features = _add_trend_feature(features,gr_period,
                                      'instalment_paid_late_in_days','{}_period_trend_'.format(period)
                                     )
        features = _add_trend_feature(features,gr_period,
                                      'instalment_paid_over_amount','{}_period_trend_'.format(period)
                                     )
    return features

def _add_trend_feature(features,gr,feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0,len(y)).reshape(-1,1)
        lr = LinearRegression()
        lr.fit(x,y)
        trend = lr.coef_[0]
    except:
        trend=np.nan
    features['{}{}'.format(prefix,feature_name)] = trend
    return features

def last_k_instalment_features_with_fractions(gr, periods, fraction_periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)

    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]

        features = add_features_in_group(features,gr_period, 'NUM_INSTALMENT_VERSION',
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))

        features = add_features_in_group(features,gr_period, 'instalment_paid_late_in_days',
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_late',
                                     ['count','mean'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_over_amount',
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period,'instalment_paid_over',
                                     ['count','mean'],
                                         'last_{}_'.format(period))

    for short_period, long_period in fraction_periods:
        short_feature_names = _get_feature_names(features, short_period)
        long_feature_names = _get_feature_names(features, long_period)

        for short_feature, long_feature in zip(short_feature_names, long_feature_names):
            old_name_chunk = '_{}_'.format(short_period)
            new_name_chunk ='_{}by{}_fraction_'.format(short_period, long_period)
            fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
            features[fraction_feature_name] = safe_div(features[short_feature], features[long_feature])
    return pd.Series(features)

def _get_feature_names(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])


def safe_div(a,b):
    try:
        return float(a)/float(b)
    except:
        return 0.0



def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features

def last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)

    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]

        features = add_features_in_group(features,gr_period, 'NUM_INSTALMENT_VERSION',
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))

        features = add_features_in_group(features,gr_period, 'instalment_paid_late_in_days',
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_late',
                                     ['count','mean'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_over_amount',
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period,'instalment_paid_over',
                                     ['count','mean'],
                                         'last_{}_'.format(period))

    return features
