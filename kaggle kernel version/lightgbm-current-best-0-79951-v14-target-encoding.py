import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
#from sklearn.impute import SimpleImputer, MICEImputer
from sklearn.preprocessing import LabelEncoder, normalize, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def feature_type_split(data, special_list=[]):
    cat_list = []
    dis_num_list = []
    num_list = []
    for i in data.columns.tolist():
        if data[i].dtype == 'object':
            cat_list.append(i)
        elif data[i].nunique() < 25:
            dis_num_list.append(i)
        elif i in special_list:     # if you want to add some special cases
            dis_num_list.append(i)
        else:
            num_list.append(i)
    return cat_list, dis_num_list, num_list

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def target_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for feature in categorical_columns:
        mapping = {}
        for col in df[feature].unique():
            mapping.update({col: df.loc[df[feature]==col, "TARGET"].mean()})
        df[feature+"_TARGET_ENCODE"] = df[feature].map(mapping)
        #df[feature].fillna(np.mean(df[feature].mean()), inplace=True)
    return df, categorical_columns

def add_target(df):
    target_info = pd.read_csv('../input/home-credit-default-risk/application_train.csv', usecols=['SK_ID_CURR','TARGET'])
    mapping = target_info.set_index('SK_ID_CURR').to_dict()['TARGET']
    del target_info; gc.collect()
    df['TARGET'] = df['SK_ID_CURR'].map(mapping)
    
    return df
    
# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = True):
    # Read data and merge
    df = pd.read_csv('../input/home-credit-default-risk/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../input/home-credit-default-risk/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    df["DOCUMENT_SEL"] = df["FLAG_DOCUMENT_4"]
    for i in [6, 7, 10, 12, 13, 14, 15, 16, 17]:
        df["DOCUMENT_SEL"] += df["FLAG_DOCUMENT_" + str(i)]
    
    drop_features = ["BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",	"YEARS_BUILD_AVG",	"COMMONAREA_AVG",	"ELEVATORS_AVG", 	"ENTRANCES_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG",	"LIVINGAPARTMENTS_AVG", 	"LIVINGAREA_AVG", 	"NONLIVINGAPARTMENTS_AVG","NONLIVINGAREA_AVG",	"APARTMENTS_MODE", 	"BASEMENTAREA_MODE",	
                 "YEARS_BEGINEXPLUATATION_MODE", "COMMONAREA_MODE",	"ELEVATORS_MODE", "ENTRANCES_MODE",	"FLOORSMIN_MODE", 	"LANDAREA_MODE",	"LIVINGAPARTMENTS_MODE",	"LIVINGAREA_MODE",	"NONLIVINGAPARTMENTS_MODE",	"APARTMENTS_MEDI",	"BASEMENTAREA_MEDI",	"YEARS_BEGINEXPLUATATION_MEDI","EMERGENCYSTATE_MODE",
                 "YEARS_BUILD_MEDI",	"COMMONAREA_MEDI",	"ELEVATORS_MEDI",	"ENTRANCES_MEDI",	"FLOORSMAX_MEDI",	"FLOORSMIN_MEDI",	"LANDAREA_MEDI",	"LIVINGAPARTMENTS_MEDI",	"LIVINGAREA_MEDI",	"NONLIVINGAPARTMENTS_MEDI",	"NONLIVINGAREA_MEDI",	"FONDKAPREMONT_MODE",	"HOUSETYPE_MODE", "WALLSMATERIAL_MODE",	
                 'FLAG_DOCUMENT_10','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_17','FLAG_DOCUMENT_19','FLAG_DOCUMENT_2','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21','FLAG_DOCUMENT_4',
                 'FLAG_DOCUMENT_7','FLAG_DOCUMENT_9','FLAG_EMP_PHONE','FLAG_MOBIL']
    df = df.drop(drop_features, axis=1)
    
    
    df,_ = target_encoder(df)
    
    # Categorical features: Binary features and One-Hot encoding
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    drop_features = ['NAME_CONTRACT_TYPE_nan','NAME_EDUCATION_TYPE_Academic degree','NAME_EDUCATION_TYPE_nan','NAME_FAMILY_STATUS_Unknown','NAME_FAMILY_STATUS_nan','NAME_HOUSING_TYPE_Co-op apartment','NAME_HOUSING_TYPE_nan','NAME_INCOME_TYPE_Businessman','NAME_INCOME_TYPE_Maternity leave','NAME_INCOME_TYPE_Pensioner',
    'NAME_INCOME_TYPE_Student','NAME_INCOME_TYPE_Unemployed','NAME_INCOME_TYPE_nan','NAME_TYPE_SUITE_Group of people','NAME_TYPE_SUITE_Other_A']
    df = df.drop(drop_features, axis=1)
    
    cat_cols = [feat for feat in cat_cols if feat not in drop_features]
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    _, dis_num_list, num_list = feature_type_split(df, special_list=[]) 
    dis_num_list.remove('TARGET')
    
    print('start impute missing value')
    #df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    #df[dis_num_list] = SimpleImputer(strategy='most_frequent').fit_transform(df[dis_num_list])
    #df[cat_cols]  = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    #df[dis_num_list]  = SimpleImputer(strategy='most_frequent').fit_transform(df[dis_num_list])
    
    # continuous 
    #df[num_list] = MICEImputer(initial_strategy='median', n_imputations=25, n_nearest_features=20, verbose=True).fit_transform(df[num_list])
    #df[num_list]  = MICEImputer(initial_strategy='median', n_imputations=25, n_nearest_features=20, verbose=True).fit_transform(df[num_list])
    
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    #df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY LENGTH'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['CONSUMER_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    
    df["EMPTY_EXT"] = df["EXT_SOURCE_1"].isnull().map({True: 1, False: 0}) + df["EXT_SOURCE_2"].isnull().map({True: 1, False: 0}) + df["EXT_SOURCE_3"].isnull().map({True: 1, False: 0})
    df["EMPTY_EXT"] = df["EMPTY_EXT"].map({0: 0.7298178, 1: 0.8178222, 2: 0.9921877})

    df["credit_to_income"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["income_to_annuity"] = df["AMT_INCOME_TOTAL"] / df["AMT_ANNUITY"]
    df["income_person_to_annuity"] = df['INCOME_PER_PERSON'] / df["AMT_ANNUITY"]
    df["income_to_price"] = df["AMT_INCOME_TOTAL"] / df["AMT_GOODS_PRICE"]
    
    df["FULL_SOURCE_1"] = df["EXT_SOURCE_1"].fillna(np.nanmedian(df["EXT_SOURCE_1"]))
    df["FULL_SOURCE_2"] = df["EXT_SOURCE_2"].fillna(np.nanmedian(df["EXT_SOURCE_2"]))
    df["FULL_SOURCE_3"] = df["EXT_SOURCE_3"].fillna(np.nanmedian(df["EXT_SOURCE_3"]))
    
    #df['EXT_ALL'] = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].mean(axis=0)
    
    df["SCORE_SPREAD"] = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].max(axis=0) - df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].min(axis=0)
    
    #df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    #df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    #df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    #df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    #df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    #df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    df['HOUR / DAY'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] / df['AMT_REQ_CREDIT_BUREAU_DAY']
    df['DAY / WEEK'] = df['AMT_REQ_CREDIT_BUREAU_DAY'] / df['AMT_REQ_CREDIT_BUREAU_WEEK']
    df['WEEK / MONTH'] = df['AMT_REQ_CREDIT_BUREAU_WEEK'] / df['AMT_REQ_CREDIT_BUREAU_MON']
    df['MONTH / QRT'] = df['AMT_REQ_CREDIT_BUREAU_MON'] / df['AMT_REQ_CREDIT_BUREAU_QRT']
    df['QRT / YEAR'] = df['AMT_REQ_CREDIT_BUREAU_QRT'] / df['AMT_REQ_CREDIT_BUREAU_YEAR']
    
    df['OVER_EXPECT_CREDIT'] = (df.AMT_CREDIT > df.AMT_GOODS_PRICE).map({False:0, True:1})
    
    df = df.drop(["FULL_SOURCE_1","FULL_SOURCE_2","FULL_SOURCE_3"], axis=1)
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True, threshold = 0.8):
    bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv', nrows = num_rows)
    
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] < -40000, 'DAYS_CREDIT_ENDDATE'] = np.nan
    bureau.loc[bureau['DAYS_CREDIT_UPDATE'] < -40000, 'DAYS_CREDIT_UPDATE'] = np.nan
    bureau.loc[bureau['DAYS_ENDDATE_FACT'] < -40000, 'DAYS_ENDDATE_FACT'] = np.nan
    bureau['AMT_CREDIT_DEBT_RATE'] = bureau.AMT_CREDIT_SUM_DEBT/(1 + bureau['AMT_CREDIT_SUM'])
    
    bureau = add_target(bureau)
    bureau, catoge_cols = target_encoder(bureau)
    bureau = bureau.drop(['TARGET'], axis=1)
    
    
    bb = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
        
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(columns= 'SK_ID_BUREAU', inplace= True)
    del bb, bb_agg
    gc.collect()
    
    bureau['ANNUITY_LENGTH'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'std'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean', 'std'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum', 'std'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'ANNUITY_LENGTH': ['min', 'max', 'mean'],
        'AMT_CREDIT_DEBT_RATE': ['max','mean']
    }
    for cat in catoge_cols:
        num_aggregations[cat+'_TARGET_ENCODE'] = ['min', 'max', 'mean','median']
        
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACT_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    
    # Bureau 0.25-year: Active credits - using only numerical aggregations
    active = bureau[(bureau['CREDIT_ACTIVE_Active'] == 1) & (bureau['DAYS_CREDIT'] >= -141)]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACT_' + e[0] + "_025Y_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    
    # Bureau 0.5-year: Active credits - using only numerical aggregations
    active = bureau[(bureau['CREDIT_ACTIVE_Active'] == 1) & (bureau['DAYS_CREDIT'] >= -282)]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACT_' + e[0] + "_05Y_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    
    # Bureau 1-year: Active credits - using only numerical aggregations
    active = bureau[(bureau['CREDIT_ACTIVE_Active'] == 1) & (bureau['DAYS_CREDIT'] >= -365)]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACT_' + e[0] + "_1Y_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    
    # Bureau 2-year: Active credits - using only numerical aggregations
    active = bureau[(bureau['CREDIT_ACTIVE_Active'] == 1) & (bureau['DAYS_CREDIT'] >= -730)]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACT_' + e[0] + "_2Y_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLS_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    # Bureau 0.25-year: Closed credits - using only numerical aggregations
    closed = bureau[(bureau['CREDIT_ACTIVE_Closed'] == 1) & (bureau['DAYS_CREDIT'] >= -141)]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLS_' + e[0] + "_025Y_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    # Bureau 0.5-year: Closed credits - using only numerical aggregations
    closed = bureau[(bureau['CREDIT_ACTIVE_Closed'] == 1) & (bureau['DAYS_CREDIT'] >= -282)]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLS_' + e[0] + "_05Y_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    # Bureau 1-year: Closed credits - using only numerical aggregations
    closed = bureau[(bureau['CREDIT_ACTIVE_Closed'] == 1) & (bureau['DAYS_CREDIT'] >= -365)]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLS_' + e[0] + "_1Y_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    # Bureau 2-year: Closed credits - using only numerical aggregations
    closed = bureau[(bureau['CREDIT_ACTIVE_Closed'] == 1) & (bureau['DAYS_CREDIT'] >= -730)]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLS_' + e[0] + "_2Y_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    del closed, closed_agg; gc.collect()
    
    # Bureau: different type of loan - using only numerical aggregations
    CREDIT_TYPE = [col for col in bureau.columns if 'CREDIT_TYPE' in col]
    for feature in CREDIT_TYPE:
        specific_loan = bureau[bureau[feature] == 1]
        spec_agg = specific_loan.groupby('SK_ID_CURR').agg(num_aggregations)
        spec_agg.columns = pd.Index([feature + "_" + e[0] + "_" + e[1].upper() for e in spec_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(spec_agg, how='left', on='SK_ID_CURR')
        del spec_agg; gc.collect()
    
    del bureau; gc.collect()
    
    feature = pd.read_csv('../input/featureselectionhomecredit/BUREAU.CSV')
    feat_num = int(len(feature['feature']) * (1 - threshold))
    feat_drop = feature.tail(feat_num)['feature'].tolist()
    feat_drop = list( set(feat_drop) - set(['NAME_CONTRACT_TYPE_Revolving loans']) )
    bureau_agg = bureau_agg.drop(feat_drop, axis=1)
    gc.collect()
    
    return bureau_agg



# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True, threshold = 0.8):
    prev = pd.read_csv('../input/home-credit-default-risk/previous_application.csv', nrows = num_rows)
    
    prev = add_target(prev)
    prev, catoge_cols = target_encoder(prev)
    prev = prev.drop(['TARGET'], axis=1)
    
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    drop_features = ['WEEKDAY_APPR_PROCESS_START_nan','FLAG_LAST_APPL_PER_CONTRACT_nan','NAME_CASH_LOAN_PURPOSE_Building a house or an annex','NAME_CASH_LOAN_PURPOSE_Business development',
        'NAME_CASH_LOAN_PURPOSE_Buying a garage','NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land','NAME_CASH_LOAN_PURPOSE_Buying a home','NAME_CASH_LOAN_PURPOSE_Buying a new car',
        'NAME_CASH_LOAN_PURPOSE_Buying a used car','NAME_CASH_LOAN_PURPOSE_Car repairs','NAME_CASH_LOAN_PURPOSE_Education','NAME_CASH_LOAN_PURPOSE_Everyday expenses','CODE_REJECT_REASON_SYSTEM',
        'NAME_CASH_LOAN_PURPOSE_Furniture','NAME_CASH_LOAN_PURPOSE_Gasification / water supply','NAME_CASH_LOAN_PURPOSE_Hobby','NAME_CASH_LOAN_PURPOSE_Journey','NAME_CASH_LOAN_PURPOSE_Refusal to name the goal',
        'NAME_CASH_LOAN_PURPOSE_Medicine','NAME_CASH_LOAN_PURPOSE_Money for a third person','NAME_CASH_LOAN_PURPOSE_Payments on other loans','NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment',
        'NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday','NAME_CASH_LOAN_PURPOSE_nan','NAME_CONTRACT_STATUS_nan','NAME_PAYMENT_TYPE_Cashless from the account of the employer','NAME_PAYMENT_TYPE_nan',
        'CODE_REJECT_REASON_VERIF','CODE_REJECT_REASON_nan','NAME_TYPE_SUITE_Group of people','NAME_CLIENT_TYPE_XNA','NAME_CLIENT_TYPE_nan','NAME_GOODS_CATEGORY_Additional Service','NAME_GOODS_CATEGORY_Animals',
        'NAME_GOODS_CATEGORY_Direct Sales','NAME_GOODS_CATEGORY_Education','NAME_GOODS_CATEGORY_Fitness','NAME_GOODS_CATEGORY_Gardening','NAME_GOODS_CATEGORY_House Construction','NAME_GOODS_CATEGORY_Insurance',
        'NAME_GOODS_CATEGORY_Medical Supplies','NAME_GOODS_CATEGORY_Medicine','NAME_GOODS_CATEGORY_Office Appliances','NAME_GOODS_CATEGORY_Other','NAME_GOODS_CATEGORY_Sport and Leisure','NAME_GOODS_CATEGORY_Tourism',
        'NAME_GOODS_CATEGORY_Vehicles','NAME_GOODS_CATEGORY_Weapon','NAME_GOODS_CATEGORY_nan','NAME_PORTFOLIO_Cars','NAME_PORTFOLIO_nan','CHANNEL_TYPE_Car dealer','CHANNEL_TYPE_Channel of corporate sales',
        'NAME_SELLER_INDUSTRY_Jewelry','NAME_SELLER_INDUSTRY_MLM partners','NAME_SELLER_INDUSTRY_Tourism','NAME_SELLER_INDUSTRY_nan']
    
    cat_cols = [feat for feat in cat_cols if feat not in drop_features]
    prev = prev.drop(drop_features, axis=1)
    del drop_features
    gc.collect()
    
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['SPREAD'] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev['SPREAD_0_1'] = (prev['AMT_APPLICATION'] > prev['AMT_CREDIT']).map({True:1, False:0})
    prev['ANNUITY_LENGTH'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    prev['CONSUMER_GOODS_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_GOODS_PRICE']
    
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean', 'std'],
        'CNT_PAYMENT': ['mean', 'sum', 'max', 'std'],
        'APP_CREDIT_PERC': ['mean'],
        'SPREAD': ['mean'],
        'ANNUITY_LENGTH': ['mean'],
        'CONSUMER_GOODS_RATIO': ['mean']
    }
    for cat in catoge_cols:
        num_aggregations[cat+'_TARGET_ENCODE'] = ['min', 'max', 'mean', 'median']
    
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
        
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APR_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REF_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    
    
    # Previous 0.5-year Applications: Approved Applications - only numerical features
    approved = prev[(prev['NAME_CONTRACT_STATUS_Approved'] == 1) & (prev['DAYS_DECISION'] >= -282)]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APR_' + e[0] + '_05Y_' + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous 0.5-year Applications: Refused Applications - only numerical features
    refused = prev[(prev['NAME_CONTRACT_STATUS_Refused'] == 1) & (prev['DAYS_DECISION'] >= -282)]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REF_' + e[0] + '_05Y_' + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    
    
    # Previous 1-year Applications: Approved Applications - only numerical features
    approved = prev[(prev['NAME_CONTRACT_STATUS_Approved'] == 1) & (prev['DAYS_DECISION'] >= -365)]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APR_' + e[0] + '_1Y_' + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous 1-year Applications: Refused Applications - only numerical features
    refused = prev[(prev['NAME_CONTRACT_STATUS_Refused'] == 1) & (prev['DAYS_DECISION'] >= -365)]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REF_' + e[0] + '_1Y_' + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    
    
    # Previous 2-year Applications: Approved Applications - only numerical features
    approved = prev[(prev['NAME_CONTRACT_STATUS_Approved'] == 1) & (prev['DAYS_DECISION'] >= -730)]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APR_' + e[0] + '_2Y_' + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous 2-year Applications: Refused Applications - only numerical features
    refused = prev[(prev['NAME_CONTRACT_STATUS_Refused'] == 1) & (prev['DAYS_DECISION'] >= -730)]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REF_' + e[0] + '_2Y_' + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    
    
    # Previous 3-year Applications: Approved Applications - only numerical features
    approved = prev[(prev['NAME_CONTRACT_STATUS_Approved'] == 1) & (prev['DAYS_DECISION'] >= -1095)]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APR_' + e[0] + '_3Y_' + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous 3-year Applications: Refused Applications - only numerical features
    refused = prev[(prev['NAME_CONTRACT_STATUS_Refused'] == 1) & (prev['DAYS_DECISION'] >= -1095)]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REF_' + e[0] + '_3Y_' + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    
    feature = pd.read_csv('../input/featureselectionhomecredit/PREVIOUS.CSV')
    feat_num = int(len(feature['feature']) * (1 - threshold))
    feat_drop = feature.tail(feat_num)['feature'].tolist()
    prev_agg = prev_agg.drop(feat_drop, axis=1)
    gc.collect()
    
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True, threshold=0.8):
    pos = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['max', 'sum', 'mean'],
        'CNT_INSTALMENT': ['mean', 'min'],
        'SK_DPD': ['max', 'sum', 'mean', lambda x: (x!=0).mean()],
        'SK_DPD_DEF': ['max', 'sum', 'mean', lambda x: (x!=0).mean()]
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean', 'size']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    pos_2Y_agg = pos[pos['MONTHS_BALANCE'] >= -24].groupby('SK_ID_CURR').agg(aggregations)
    pos_2Y_agg.columns = pd.Index(['POS_' + e[0] + "_2Y_" + e[1].upper() for e in pos_2Y_agg.columns.tolist()])
    pos_agg = pos_agg.join(pos_2Y_agg)
    del pos_2Y_agg
    
    pos_1Y_agg = pos[pos['MONTHS_BALANCE'] >= -12].groupby('SK_ID_CURR').agg(aggregations)
    pos_1Y_agg.columns = pd.Index(['POS_' + e[0] + "_1Y_" + e[1].upper() for e in pos_1Y_agg.columns.tolist()])
    pos_agg = pos_agg.join(pos_1Y_agg)
    del pos_1Y_agg
    
    pos_05Y_agg = pos[pos['MONTHS_BALANCE'] >= -6].groupby('SK_ID_CURR').agg(aggregations)
    pos_05Y_agg.columns = pd.Index(['POS_' + e[0] + "_05Y_" + e[1].upper() for e in pos_05Y_agg.columns.tolist()])
    pos_agg = pos_agg.join(pos_05Y_agg)
    del pos_05Y_agg
    
    drop_features = ['POS_NAME_CONTRACT_STATUS_Canceled_MEAN','POS_NAME_CONTRACT_STATUS_Completed_SIZE','POS_NAME_CONTRACT_STATUS_Demand_MEAN','POS_NAME_CONTRACT_STATUS_Demand_SIZE','POS_NAME_CONTRACT_STATUS_Returned to the store_SIZE',
    'POS_NAME_CONTRACT_STATUS_Signed_SIZE','POS_NAME_CONTRACT_STATUS_XNA_MEAN','POS_NAME_CONTRACT_STATUS_XNA_SIZE','POS_NAME_CONTRACT_STATUS_nan_MEAN','POS_NAME_CONTRACT_STATUS_nan_SIZE','POS_NAME_CONTRACT_STATUS_Amortized debt_1Y_MEAN',
    'POS_NAME_CONTRACT_STATUS_Approved_1Y_MEAN','POS_NAME_CONTRACT_STATUS_Canceled_1Y_MEAN','POS_NAME_CONTRACT_STATUS_Completed_1Y_SIZE','POS_NAME_CONTRACT_STATUS_Demand_1Y_MEAN','POS_NAME_CONTRACT_STATUS_Demand_1Y_SIZE',
    'POS_NAME_CONTRACT_STATUS_Returned to the store_1Y_MEAN','POS_NAME_CONTRACT_STATUS_Returned to the store_1Y_SIZE','POS_NAME_CONTRACT_STATUS_Signed_1Y_SIZE','POS_NAME_CONTRACT_STATUS_XNA_1Y_MEAN','POS_NAME_CONTRACT_STATUS_XNA_1Y_SIZE',
    'POS_NAME_CONTRACT_STATUS_nan_1Y_MEAN','POS_NAME_CONTRACT_STATUS_nan_1Y_SIZE','POS_NAME_CONTRACT_STATUS_Amortized debt_05Y_MEAN','POS_NAME_CONTRACT_STATUS_Approved_05Y_MEAN','POS_NAME_CONTRACT_STATUS_Canceled_05Y_MEAN',
    'POS_NAME_CONTRACT_STATUS_Canceled_05Y_SIZE','POS_NAME_CONTRACT_STATUS_Demand_05Y_MEAN','POS_NAME_CONTRACT_STATUS_Demand_05Y_SIZE','POS_NAME_CONTRACT_STATUS_Returned to the store_05Y_MEAN','POS_NAME_CONTRACT_STATUS_Returned to the store_05Y_SIZE',
    'POS_NAME_CONTRACT_STATUS_Signed_05Y_SIZE','POS_NAME_CONTRACT_STATUS_XNA_05Y_MEAN','POS_NAME_CONTRACT_STATUS_XNA_05Y_SIZE','POS_NAME_CONTRACT_STATUS_nan_05Y_MEAN','POS_NAME_CONTRACT_STATUS_nan_05Y_SIZE']
    
    pos_agg = pos_agg.drop(drop_features, axis=1)
    del pos
    
    
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True, threshold=0.8):
    ins = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_PERC'] = 1 - ins['PAYMENT_PERC']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique', 'sum'],
        'DPD': ['max', 'mean', 'sum', 'std', lambda x: (x!=0).mean()],
        'DBD': ['max', 'mean', 'sum', 'std', lambda x: (x!=0).mean()],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'std'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum', 'std'],
        'AMT_PAYMENT': ['min', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    
    ins_2Y_agg = ins[ins['DAYS_INSTALMENT'] >= -730].groupby('SK_ID_CURR').agg(aggregations)
    ins_2Y_agg.columns = pd.Index(['INS_' + e[0] + "_2Y_" + e[1].upper() for e in ins_2Y_agg.columns.tolist()])
    ins_agg = ins_agg.join(ins_2Y_agg)
    del ins_2Y_agg
    
    ins_1Y_agg = ins[ins['DAYS_INSTALMENT'] >= -365].groupby('SK_ID_CURR').agg(aggregations)
    ins_1Y_agg.columns = pd.Index(['INS_' + e[0] + "_1Y_" + e[1].upper() for e in ins_1Y_agg.columns.tolist()])
    ins_agg = ins_agg.join(ins_1Y_agg)
    del ins_1Y_agg
    
    ins_05Y_agg = ins[ins['DAYS_INSTALMENT'] >= -182].groupby('SK_ID_CURR').agg(aggregations)
    ins_05Y_agg.columns = pd.Index(['INS_' + e[0] + "_05Y_" + e[1].upper() for e in ins_05Y_agg.columns.tolist()])
    ins_agg = ins_agg.join(ins_05Y_agg)
    del ins_05Y_agg
    
    ins_025Y_agg = ins[ins['DAYS_INSTALMENT'] >= -91].groupby('SK_ID_CURR').agg(aggregations)
    ins_025Y_agg.columns = pd.Index(['INS_' + e[0] + "_025Y_" + e[1].upper() for e in ins_025Y_agg.columns.tolist()])
    ins_agg = ins_agg.join(ins_025Y_agg)
    del ins_025Y_agg; gc.collect()
    
    
    # Count installments accounts
    ins_agg['INS_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins; gc.collect()
    
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True, threshold=0.8):
    cc = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv', nrows = num_rows)
    
    cc.loc[cc['AMT_DRAWINGS_ATM_CURRENT'] < 0, 'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
    cc.loc[cc['AMT_DRAWINGS_CURRENT'] < 0, 'AMT_DRAWINGS_CURRENT'] = np.nan
    
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(columns = ['SK_ID_PREV'], inplace = True)
    cc['BALANCE_CREDIT_RATIO'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['REQUIRED_PAYMENT_GAP'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']
    aggregations = {
        'SK_DPD': ['max', 'sum', 'mean', lambda x: (x!=0).mean()],
        'SK_DPD_DEF': ['max', 'sum', 'mean', lambda x: (x!=0).mean()],
        'AMT_BALANCE': ['min', 'max', 'mean', 'var'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'var'],
        'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'var'],
        'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'var'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'var'],
        'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'var'],
        'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean', 'var'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'var'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'var'],
        'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'var'],
        'AMT_RECIVABLE': ['min', 'max', 'mean', 'var'],
        'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'var'],
        'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'var'],
        'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'var'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'var'],
        'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'var'],
        'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'var'],
        'BALANCE_CREDIT_RATIO': ['mean', 'std'],
        'REQUIRED_PAYMENT_GAP': ['mean', 'min', 'std']
    }
    cc_agg = cc.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    
    cc_05Y_agg = cc[cc['MONTHS_BALANCE'] >= -6].groupby('SK_ID_CURR').agg(aggregations)
    cc_05Y_agg.columns = pd.Index(['CC_' + e[0] + "_05Y_" + e[1].upper() for e in cc_05Y_agg.columns.tolist()])
    cc_agg = cc_agg.join(cc_05Y_agg)
    del cc_05Y_agg
    
    cc_1Y_agg = cc[cc['MONTHS_BALANCE'] >= -12].groupby('SK_ID_CURR').agg(aggregations)
    cc_1Y_agg.columns = pd.Index(['CC_' + e[0] + "_1Y_" + e[1].upper() for e in cc_1Y_agg.columns.tolist()])
    cc_agg = cc_agg.join(cc_1Y_agg)
    del cc_1Y_agg
    
    cc_2Y_agg = cc[cc['MONTHS_BALANCE'] >= -24].groupby('SK_ID_CURR').agg(aggregations)
    cc_2Y_agg.columns = pd.Index(['CC_' + e[0] + "_2Y_" + e[1].upper() for e in cc_2Y_agg.columns.tolist()])
    cc_agg = cc_agg.join(cc_2Y_agg)
    del cc_2Y_agg
    
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    
    return cc_agg


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, submission_file_name, stratified = False):
    
    #df['LIMIT_INCOME_RATIO'] = df['CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN'] / df['AMT_INCOME_TOTAL']
    #df['MIN_PAYMENT_INCOME_RATIO'] = df['CC_AMT_INST_MIN_REGULARITY_MEAN'] / df['AMT_INCOME_TOTAL']
    #df = df.drop(["EXT_ALL"], axis=1)
    
    
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    categorical = [f for f in df.columns if (f not in ['TARGET']) & (df[f].max()==1) & (df[f].min()==0) & ('_MEAN' not in f) ]
    num_features = [item for item in feats if item not in categorical]
    
    
    
    #df[num_features] = df[num_features].transform(lambda x : np.clip(x,x.quantile(0.01),x.quantile(0.99)))
    #df[num_features] = df[num_features].transform(lambda x : np.clip(x,np.nanpercentile(x, 0.01, axis=0),np.nanpercentile(x, 0.99, axis=0)))
    
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    #del train_df_positive
    gc.collect()
    
    # Cross validation model
    # seem like a good random seed : 363, 5197, 371, 2420, 7535, 5407
    rand_seed = np.random.randint(0,9999)
    print('Split random seed: ', rand_seed)
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=5407)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=5407)
    
    #np.random.randint(0,9999)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = feats
    feature_importance_df["importance"] = [0] * len(feats)
    
    
    fold_auc = []
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        
        
        method = ''
        
        if method == 'dart':
            
            # LightGBM parameters found by Bayesian optimization
            clf = LGBMClassifier(
                boosting_type='dart',
                drop_rate= 0.02,
                nthread=4,
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=34,
                colsample_bytree=0.7,
                subsample=0.8715623,
                max_depth=-1,
                reg_alpha=10,
                #reg_lambda=10,
                min_split_gain=0.0222415,
                min_child_weight=39.3259775,
                silent=-1,
                verbose=-1,
            )
            
        else:
            
            clf = LGBMClassifier(
                nthread=4,
                n_estimators=10000,
                learning_rate=0.01,
                num_leaves=34,
                colsample_bytree=0.7497036,
                subsample=0.8715623,
                max_depth=9,
                reg_alpha=10,
                reg_lambda=0.0735294,
                min_split_gain=0.0222415,
                min_child_weight=39.3259775,
                silent=-1,
                verbose=-1,
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric= 'auc', verbose= 800, early_stopping_rounds= 400)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        feature_importance_df["importance"] = feature_importance_df["importance"] + clf.feature_importances_

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        fold_auc.append(roc_auc_score(valid_y, oof_preds[valid_idx]))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    train_df['TRAIN_SCORE'] = oof_preds
    train_df[['SK_ID_CURR', 'TRAIN_SCORE']].to_csv('all_feature_TRAIN.csv', index=False)
    print('Average AUC score %.6f' % np.mean(fold_auc))
    print('standard deviation %.6f' % np.std(fold_auc))
    # Write submission file and plot feature importance
    test_df['TARGET'] = sub_preds
    test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype('uint32').values
    test_df[['SK_ID_CURR', 'TARGET']].to_csv("all_feature_TEST.csv", index=False)
    
    feature_importance_df["importance"] = feature_importance_df["importance"] / n_fold
    feature_importance_df = feature_importance_df.sort_values(by=['importance'], ascending=False)
    feature_importance_df.to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-01.png')


def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows,threshold = 0.6)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows,threshold = 0.6)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
        df = reduce_mem_usage(df)
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        
        threshold = 0.6
        feature = pd.read_csv('../input/featureselectionhomecredit/REST.CSV')
        feat_num = int(len(feature['feature']) * (1 - threshold))
        feat_drop = feature.tail(feat_num)['feature'].tolist()
        feat_drop = [feat for feat in feat_drop if feat in df.columns]
        df = df.drop(feat_drop, axis=1)
        del feature, feat_num, feat_drop, threshold; gc.collect()
        
    with timer("Run LightGBM with kfold"):
        df = reduce_mem_usage(df)

        threshold = 0.8
        feature = pd.read_csv('../input/featureselectionhomecredit/all_features.csv')
        feat_num = int(len(feature['feature']) * (1 - threshold))
        feat_drop = feature.tail(feat_num)['feature'].tolist()
        feat_drop = [feat for feat in feat_drop if feat in df.columns]
        df = df.drop(feat_drop, axis=1)
        del feature, feat_num, feat_drop, threshold; gc.collect()
        

        threshold = 0.75
        feature = pd.read_csv('../input/featureselectionhomecredit/all_features_v2.CSV')
        feat_num = int(len(feature['feature']) * (1 - threshold))
        feat_drop = feature.tail(feat_num)['feature'].tolist()
        feat_drop = [feat for feat in feat_drop if feat in df.columns]
        df = df.drop(feat_drop, axis=1)
        del feature, feat_num, feat_drop, threshold; gc.collect()
        
        threshold = 0.34
        feature = pd.read_csv('../input/featureselectionhomecredit/all_features_v3.csv')
        feat_num = int(len(feature['feature']) * (1 - threshold))
        feat_drop = feature.tail(feat_num)['feature'].tolist()
        feat_drop = [feat for feat in feat_drop if feat in df.columns]
        df = df.drop(feat_drop, axis=1)
        del feature, feat_num, feat_drop, threshold; gc.collect()
        
        feat_importance = kfold_lightgbm(df,num_folds=5,submission_file_name='all_features_importance.csv',stratified=False)
    
        
if __name__ == "__main__":
    with timer("Full model run"):
        main(debug= False)