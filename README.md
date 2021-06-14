# predict-future-sales

## Run ##


### 安裝 <h3> 
```
pip install -r requirements.txt
```

### 執行 <h3> 
```
python main.py 
```
  
執行main.py將訓練4個模型(GRU , XGBoost , LightBGM ,linear regression)
並output出一個submission.csv
  
執行時間：1341.848888 秒

  
## Data ##

### Download <h3>
```
https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data
```  
  
資料集為2013年1月至2015年10月每日銷售數據，
  
共有60間商店、22,171項不同的商品，
  
產品分為84種類型，
  
需要預測的項目為2015年11月商店名稱與商品名稱，
  
根據對應的商店與商品預測其此月份的銷售數量。

## 資料整理 ##
  
所有商品的每月銷售總數量:
  
![image](https://user-images.githubusercontent.com/66662065/121816256-29586700-ccad-11eb-9f9d-9c1094a41236.png)

找出已停售的商品,若test.csv有出現相同 item_ID則直接將月銷售設為0:

![image](https://user-images.githubusercontent.com/66662065/121816237-104fb600-ccad-11eb-9e24-5e789fae1e20.png)

找出已倒閉的店家,若test.csv有出現相同 shop_ID則直接將月銷售設為0:

![image](https://user-images.githubusercontent.com/66662065/121816245-19d91e00-ccad-11eb-92a4-8f25aed16200.png)

刪除離群值:

![image](https://user-images.githubusercontent.com/66662065/121816269-35442900-ccad-11eb-9c80-24cb58800606.png)

更改異常的價格至同商店商品的中位數:
  
![image](https://user-images.githubusercontent.com/66662065/121816283-49882600-ccad-11eb-8baa-9b39bfd51394.png)



## Feature ##

### 前1~6月的特徵 ###
  
。商店商品銷售數量平均、總和
  
。每件商品銷售量平均
  
。每個商店的銷售量平均
  
。每個商品類別銷售量平均
  
。每個商店類型銷售量平均
  
。每個城市銷售量平均
  
。每個城市及商店類型銷售量平均
  

### 售量趨勢 ###

。上個月的銷售量 – 上上個月銷售量

### 日期 ###
  
。當下年份、月份

。當月的天數

## Model ##

train data : 前 32 個月的data

valid data : 第 33 個月的data

test data : 第 34 個月的

  
![image](https://user-images.githubusercontent.com/66662065/121887838-d7b1ea00-cd49-11eb-960f-68752d4cb6ff.png)

  

  

## First Stage ##

### **LightGBM** <h3>

training's rmse: 0.590989	valid_1's rmse: 0.684242

  
### **GRU** <h3>

loss: 0.5016 - val_loss: 0.5839


### **XGBoost** <h3>
  
validation_0-rmse:0.60311	validation_1-rmse:0.68829
  
### **Linear Regression** <h3>
  
## Second Stage ##  

將4個模型預測"valid-data"整合成First Stage資料集

將資料丟進Linear Regression做第二步的訓練
  
並使用訓練好的Model來做test-data的預測
  
![image](https://user-images.githubusercontent.com/66662065/121899100-ac81c780-cd56-11eb-8bff-3445950eda95.png)


## GOOGLE 簡報 ##

https://docs.google.com/presentation/d/10lyV-RkyssigZESbjqtZ_Od4YDNGMGI87b_R4IXjdYw/edit?usp=sharing
