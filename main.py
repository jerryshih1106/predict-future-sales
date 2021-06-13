from sklearn.datasets import load_iris
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
import func
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.linear_model import LinearRegression

test = pd.read_csv('test.csv', dtype={'ID': 'int32', 'shop_id': 'int32','item_id': 'int32'})
item_categories = pd.read_csv('item_categories.csv',dtype={'item_category_name': 'str', 'item_category_id': 'int32'})
items = pd.read_csv('items.csv', dtype={'item_name': 'str', 'item_id': 'int32','item_category_id': 'int32'})
shops = pd.read_csv('shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
sales = pd.read_csv('sales_train.csv', parse_dates=['date'],dtype={'date': 'str', 'date_block_num': 'int32', 
                                                                    'shop_id': 'int32','item_id': 'int32', 'item_price': 'float32',
                                                                    'item_cnt_day': 'int32'})
#=============================================建造===================================================================================|||
    
test = pd.read_csv('test.csv', dtype={'ID': 'int32', 'shop_id': 'int32','item_id': 'int32'})
item_categories = pd.read_csv('item_categories.csv',dtype={'item_category_name': 'str', 'item_category_id': 'int32'})
items = pd.read_csv('items.csv', dtype={'item_name': 'str', 'item_id': 'int32','item_category_id': 'int32'})
shops = pd.read_csv('shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
sales = pd.read_csv('sales_train.csv', parse_dates=['date'],dtype={'date': 'str', 'date_block_num': 'int32', 
                                                                    'shop_id': 'int32','item_id': 'int32', 'item_price': 'float32',
                                                                    'item_cnt_day': 'int32'})
#============處理資料
sales = sales[sales['item_cnt_day'] <=1000]#只留<=1000的
sales = sales[sales['item_cnt_day'] >0]#1.12
sales = sales[sales['item_price'] <= 30000]
median = sales[(sales['shop_id'] == 32) & (sales['item_id'] == 2973) & (sales['item_price']>0)].item_price.median()
sales.loc[sales['item_price']<0,'item_price'] = median#此商品所有的價格的中位數
#====================
# #————————————————將全部資料整合成一個
train = sales.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)
# ————————————————
All_Tshop_id = test['shop_id'].unique()
All_Titem_id = test['item_id'].unique()
lk_train = train[train['shop_id'].isin(All_Tshop_id)]
lk_train = lk_train[lk_train['item_id'].isin(All_Titem_id)]
train_mon = lk_train[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]
train_mon.head().append(train_mon.tail())
#————————————————
train_mon = train_mon.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id'], as_index=False)
# new_item = train_mon
#==============================
train_mon = train_mon.agg({'item_price':['sum','mean'], 'item_cnt_day':['sum', 'mean','count']})
train_mon.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price',
                          'item_price_mean','item_cnt_sum','item_cnt_mean', 'item_cnt_month']
train_mon.head().append(train_mon.tail())
#————————————————
shop_ids = train_mon['shop_id'].unique()
item_ids = train_mon['item_id'].unique()
matrix = []
for i in range(34):
    for shop in shop_ids:
        for item in item_ids:
            matrix.append([shop, item,i])
matrix = pd.DataFrame(matrix, columns=['shop_id','item_id','date_block_num'])
train_mon = pd.merge(matrix, train_mon, on=['shop_id','item_id','date_block_num'], how='left')
train_mon.fillna(0, inplace=True)
# #====================新增的column==========================================
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
#===========================train test合===============================================
cols = ['date_block_num','shop_id','item_id']
train_mon = pd.concat([train_mon, test[['item_id','shop_id','date_block_num']]], ignore_index=True, sort=False, keys=cols)
train_mon.fillna(0, inplace=True) # 34 month
#====================================feature==============================================
train_mon = func.build_lag_feature(train_mon, [1,2,3,4,5,6,12], 'item_cnt_month')
#===================商店中的商品每個月的銷售量
group = train_mon.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = ['avg_item_cnt']
group.reset_index(inplace=True)
train_mon = pd.merge(train_mon, group, on=['date_block_num'], how='left')
train_mon['avg_item_cnt'] = train_mon['avg_item_cnt'].astype(np.float16)
train_mon = func.build_lag_feature(train_mon, [1,2,3,4,5,6,12], 'avg_item_cnt')
train_mon.drop(['avg_item_cnt'], axis=1, inplace=True)
#===================每個商品每個月的銷售量
group = train_mon.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['item_avg_item_cnt']
group.reset_index(inplace=True)
train_mon = pd.merge(train_mon, group, on=['date_block_num','item_id'], how='left')
train_mon['item_avg_item_cnt'] = train_mon['item_avg_item_cnt'].astype(np.float16)
train_mon = func.build_lag_feature(train_mon, [1,2,3,4,5,6,12], 'item_avg_item_cnt')
train_mon.drop(['item_avg_item_cnt'], axis=1, inplace=True)
#===================每間商店每個月的銷售量
group = train_mon.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['shop_avg_item_cnt']
group.reset_index(inplace=True)
train_mon = pd.merge(train_mon, group, on=['date_block_num','shop_id'], how='left')
train_mon['shop_avg_item_cnt'] = train_mon['shop_avg_item_cnt'].astype(np.float16)
train_mon =func.build_lag_feature(train_mon, [1,2,3,4,5,6,12], 'shop_avg_item_cnt')
train_mon.drop(['shop_avg_item_cnt'], axis=1, inplace=True)
#+==================每個月商品類別銷售量
group = train_mon.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['cat_item_cnt']
group.reset_index(inplace=True)
train_mon = pd.merge(train_mon, group, on=['date_block_num','item_category_id'], how='left')
train_mon['cat_item_cnt'] = train_mon['cat_item_cnt'].astype(np.float16)
train_mon = func.build_lag_feature(train_mon, [1,2,3,4,5,6,12], 'cat_item_cnt')
train_mon.drop(['cat_item_cnt'], axis=1, inplace=True)
#+==================每個月商店商品類別銷售量
group = train_mon.groupby(['date_block_num', 'item_category_id','shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_cat_shop_avg_item_cnt' ]
group.reset_index(inplace=True)
train_mon = pd.merge(train_mon, group, on=['date_block_num', 'item_category_id','shop_id'], how='left')
train_mon['date_cat_shop_avg_item_cnt'] = train_mon['date_cat_shop_avg_item_cnt'].astype(np.float16)
train_mon = func.build_lag_feature(train_mon, [1,2,3,4,5,6,12], 'date_cat_shop_avg_item_cnt')
train_mon.drop(['date_cat_shop_avg_item_cnt'], axis=1, inplace=True)

#+==================每個月天數
# train_mon['month'] = train_mon['date_block_num'] % 12
train_mon['year'] = train_mon['date_block_num'].apply(lambda x: ((x//12) + 2013))
train_mon['month'] = train_mon['date_block_num'].apply(lambda x: (x % 12))
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
train_mon['days'] = train_mon['month'].map(days).astype(np.int8)
# train_mon['month'] = train_mon['date_block_num'].apply(lambda x: (x % 12 + 1))

#==========================TREND=================================
train_mon["trend"] = train_mon["item_cnt_month_lag_1"] - train_mon["item_cnt_month_lag_2"]
#================================================================
#===================lag6個月,消除前6個月資訊
print(train_mon.isna().sum())
train_mon = train_mon[train_mon.date_block_num >11]
train_mon = func.fill_nan(train_mon)
# train_mon.fillna(0)
train_mon["trend"].fillna(0, inplace=True) 
print(train_mon.isna().sum())
# sales = sales[sales['item_price'] ]
#==========================================================================model======================================================================================================
# data = pd.read_csv('123.csv')
data = train_mon
dataM1 = data['item_cnt_month']
# dataM = np.resize(dataM1,(4569012,1))
dataM1 = pd.DataFrame(dataM1)
data = data.drop(['item_category_id', 'item_id', 'item_price','item_price_mean','item_cnt_sum','item_cnt_mean','item_cnt_month'], axis=1)

scaler = StandardScaler().fit(data)
data['item_cnt_month'] = dataM1['item_cnt_month']
Lscaler = StandardScaler().fit(dataM1)
data['item_cnt_month'].fillna(data['item_cnt_month'].median(),inplace = True)
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)#接給xgboost lightgbm
nor_X_train = scaler.transform(X_train)
nor_X_train = np.resize(nor_X_train,(len(X_train),1,48))
#----
Y_train = dataM1[data.date_block_num < 33]['item_cnt_month']
Y_train = np.resize(Y_train,(len(X_train),1))
Y_train = pd.DataFrame(Y_train)#接給xgboost lightgbm
nor_Y_train = Lscaler.transform(Y_train)
nor_Y_train = np.resize(nor_Y_train,(len(Y_train),1))

#----
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)#接給xgboost lightgbm
nor_X_valid = scaler.transform(X_valid)
nor_X_valid = np.resize(nor_X_valid,(len(X_valid),1,48))
#----
Y_valid = dataM1[data.date_block_num == 33]['item_cnt_month']
Y_valid = np.resize(Y_valid,(len(X_valid),1))
Y_valid = pd.DataFrame(Y_valid)#接給xgboost lightgbm
nor_Y_valid = Lscaler.transform(Y_valid)
nor_Y_valid = np.resize(nor_Y_valid,(len(Y_valid),1))

#----
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)#接給xgboost lightgbm
nor_X_test = scaler.transform(X_test)
nor_X_test = np.resize(nor_X_test,(len(X_test),1,48))


#====================GRU========================
GRU_model = func.buildModel(nor_X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
GRU_model.fit(nor_X_train, nor_Y_train, epochs=20, batch_size=160, validation_data=(nor_X_valid, nor_Y_valid), callbacks=[callback])
#=============save===========
GRU_model.save('main_GRU_model.h5')
# GRU_model = load_model('main_GRU_model.h5')
#=============save===========
GRU_Y_test = GRU_model.predict(nor_X_test)
GRU_Y_test = np.resize(GRU_Y_test,(len(X_test),1))
GRU_Y_test = pd.DataFrame(GRU_Y_test)
GRU_Y_test = Lscaler.inverse_transform(GRU_Y_test)
GRU_Y_test = GRU_Y_test.clip(0, 20)
GRU_Y_test = pd.DataFrame(GRU_Y_test)
#=================================XGBOOST=================================
xgb_model = XGBRegressor(max_depth=10, 
                          n_estimators=1000, 
                          min_child_weight=500,  
                          colsample_bytree=0.8, 
                          subsample=0.8, 
                          eta=0.3, 
                          seed=0)
xgb_model.fit(X_train, 
              Y_train, 
              eval_metric="rmse", 
              eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
              verbose=20, 
              early_stopping_rounds=20)
#=============save===========
joblib.dump(xgb_model, 'main_XGB_model.pkl')
# xgb_model = joblib.load('main_XGB_model.pkl')
#=============save===========
xgb_Y_test = xgb_model.predict(X_test)
# xgb_Y_test = np.resize(xgb_Y_test,(len(X_test),1))
# xgb_Y_test = pd.DataFrame(xgb_Y_test).clip(0, 20)

#===========================lightgbm
import lightgbm as lgb
train_data = lgb.Dataset(data=X_train, label=Y_train)
valid_data = lgb.Dataset(data=X_valid, label=Y_valid)
params = {"objective" : "regression", "metric" : "rmse", 'n_estimators':100000, 'early_stopping_rounds':50,
              "num_leaves" : 200, "learning_rate" : 0.01, "bagging_fraction" : 0.9,
              "feature_fraction" : 0.3, "bagging_seed" : 0}
lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000) 
#=============save===========
joblib.dump(lgb_model, 'main_LGB_model.pkl')
# lgb_model = joblib.load('main_LGB_model.pkl')
#=============save===========
lgb_Y_test = lgb_model.predict(X_test)
# lgb_Y_test = np.resize(lgb_Y_test,(len(X_test),1))
# lgb_Y_test = pd.DataFrame(lgb_Y_test).clip(0, 20)
#================================================
# Y_test = []
# Y_test = pd.DataFrame()
# Y_test['item_cnt_month'] = GRU_Y_test[0]*2 +lgb_Y_test[0] +xgb_Y_test[0]*3 
# Y_test['item_cnt_month'] = Y_test['item_cnt_month']/6
#==============================2stage===============================================
xgb_Y_valid = xgb_model.predict(X_valid)
lgb_Y_valid = lgb_model.predict(X_valid)
GRU_Y_valid = GRU_model.predict(nor_X_valid)
GRU_Y_valid = np.resize(GRU_Y_valid,(len(X_valid),1))
GRU_Y_valid = pd.DataFrame(GRU_Y_valid)
GRU_Y_valid= Lscaler.inverse_transform(GRU_Y_valid)

first_level = pd.DataFrame()
first_level['lgb'] = lgb_Y_valid
first_level['xgbm'] = xgb_Y_valid
first_level['GRU'] = GRU_Y_valid

first_level_test = pd.DataFrame()
first_level_test['lgb'] = lgb_Y_test
first_level_test['xgbm'] = xgb_Y_test
first_level_test['GRU'] = GRU_Y_test
meta_model = LinearRegression(n_jobs=-1)
meta_model.fit(first_level, Y_valid)
ensemble_pred = meta_model.predict(first_level)
final_predictions = meta_model.predict(first_level_test)

#=============================================================================
sale = sales[sales['date_block_num'] == 33]
grouped = sale[['shop_id','item_id','item_cnt_day']].groupby(['shop_id','item_id']).agg({'item_cnt_day':'sum'}).reset_index()
grouped = grouped.rename(columns={'item_cnt_day' : 'item_cnt_month'})
test = pd.merge(test,grouped, on = ['shop_id','item_id'], how = 'left')
# test['item_cnt_month'] = test['item_cnt_month'].fillna(0).clip(0,20) #新增每月銷量
test['item_cnt_month'] = final_predictions
#==============================================一年沒賣出商品=========================================
sales_by_item_id = sales.pivot_table(index=['item_id'],values=['item_cnt_day'],columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'
outdated_items = sales_by_item_id[sales_by_item_id.loc[:,'21':].sum(axis=1)==0] #連續一年都沒有賣出商品
outlist = test[test['item_id'].isin(outdated_items['item_id'])]['item_id'].unique()
test.fillna(0, inplace=True)
# print(test.isna().sum())
for i in range(len(outlist)):    
    test.loc[test['item_id'] == outlist[i] ,'item_cnt_month'] = 0

closelist =  [ 0 , 1 , 8, 11, 13 ,17 ,23 ,29 ,30 ,32, 33 ,40 ,43, 54]
closelist = np.array(closelist)
for i in range(len(closelist)):
    test.loc[test['shop_id'] == closelist[i] ,'item_cnt_month'] = 0

#=====================================================================================================

test = test[['ID','item_cnt_month']]
submission = test.set_index('ID')
submission.to_csv('submission.csv')