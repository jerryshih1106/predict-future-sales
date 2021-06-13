from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, XGBRegressor
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from tensorflow.keras import regularizers
import joblib


def buildModel(shape):
    model = Sequential()
    # model.add(GRU(50,input_length=shape[1], input_dim=shape[2],return_sequences=True))
    # model.add(GRU(3,input_length=shape[1], input_dim=shape[2],return_sequences=True))
    model.add(GRU(5,input_length=shape[1], input_dim=shape[2]))
    # model.add(Dropout(0.1))
    # model.add(GRU(5,return_sequences=True))
    # model.add(Dropout(0.1))
    # model.add(GRU(7,return_sequences=True))
    # model.add(Dropout(0.1))
    # model.add(GRU(10))
    model.add(Dropout(0.1))
    # model.add(GRU(5))
    # model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def build_lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

def fill_nan(df):#取代na為零
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df
