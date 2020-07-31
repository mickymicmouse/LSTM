# -*- coding: euc-kr -*-
"""
Created on Thu Jul 30 10:06:00 2020

@author: seungjun
"""
# 마켓별 상장 기업 리스트 불러오기
import matplotlib.pyplot as plt
import dart_fss as dart
import pandas_datareader.data as web
import pandas as pd
import sqlite3
import datetime
import urllib.parse
import pandas_datareader as pdr
import numpy as np
import os
from io import BytesIO 
from urllib import request 
import pandas as pd 
req_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx?' \
     'name=fileDown' \ 
     '&filetype=xls' \
     '&url=MKD/04/0406/04060100/mkd04060100_01' \
     '&market_gubun=STK' \
     '&isu_cdnm=%EC%A0%84%EC%B2%B4' \
     '&sort_type=A' \ 
     '&lst_stk_vl=1' \ 
     '&isu_cdnm=%EC%A0%84%EC%B2%B4' \
     '&pagePath=%2Fcontents%2FMKD%2F04%2F0406%2F04060100%2FMKD04060100.jsp' 
req = request.Request(req_url) 
code = request.urlopen(req).read().decode() 
req_url = 'http://file.krx.co.kr/download.jspx?code=' + code 
headers = {'Referer': 'http://marketdata.krx.co.kr/mdi'} 
req = request.Request(req_url, headers=headers) 
# 헤더(리퍼러) 추가 
byte_data = request.urlopen(req).read() 
kospi_stock = pd.read_excel(BytesIO(byte_data))


kospi_stock.head()



results = {}
for code in kospi_stock.종목코드[:7]:
    results[code] = pdr.DataReader(code + '.KS', 'yahoo', '2020-01-01', None)

df = pd.concat(results, axis=1)
df.loc[:, pd.IndexSlice[:, 'Adj Close']].tail()

code = kospi_stock.종목코드[kospi_stock.회사명 == "삼성전자"]
samsung = pdr.DataReader(code + '.KS', 'yahoo', '2014-12-31', None)
samsung.index

pd.to_datetime(samsung.index, format = '%Y%m%d')
samsung['일자'] = pd.to_datetime(samsung.index, format='%Y%m%d')
samsung['연도'] =samsung['일자'].dt.year
samsung['월'] =samsung['일자'].dt.month
samsung['일'] =samsung['일자'].dt.day


plt.figure(figsize=(16, 9))
plt.plot( samsung['일자'],samsung['Close'])
plt.xlabel('time')
plt.ylabel('price')
plt.grid()
plt.savefig(os.path.join(model_path,"train_image"))


#Normlized
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df_scaled = scaler.fit_transform(samsung[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

print(df_scaled)


#make Dataset
TEST_SIZE=200
train = samsung[:-TEST_SIZE]
test = samsung[-TEST_SIZE:]


def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

feature_cols = ['Open','High','Low','Volume']
label_cols = ['Close']

train_feature = train[feature_cols]
train_label = train[label_cols]
test_feature = test[feature_cols]
test_label = test[label_cols]


train_feature, train_label = make_dataset(train_feature, train_label, 20)
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)



x_train.shape, x_valid.shape
# ((6086, 20, 4), (1522, 20, 4))

# test dataset (실제 예측 해볼 데이터)
test_feature, test_label = make_dataset(test_feature, test_label, 20)
test_feature.shape, test_label.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(16, 
               input_shape=(train_feature.shape[1], train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )
model.add(Dense(1))
model_path = r"/home/itm1/seungjun/LSTM"
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train, 
                    epochs=200, 
                    batch_size=16,
                    validation_data=(x_valid, y_valid), 
                    callbacks=[early_stop, checkpoint])

model.load_weights(filename)
pred = model.predict(test_feature)

plt.figure(figsize=(12, 9))
plt.plot(test_label, label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.savefig(os.path.join(model_path,"test_image"))
