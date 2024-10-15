from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing.pool
import pandas as pd
import numpy as np
import requests
from strategy.utils.utils import *
from datetime import datetime
from dateutil import rrule
import matplotlib.pyplot as plt
import random
import multiprocessing
import pandas as pd
import time

def generate_data(future):
    data =pd.read_excel(f"data/{future}_daily.xlsx",index_col=0)
    # 
    data["sigma"]=data["profit"].rolling(window=63).std()
    data["profit20"]=data["profit"].rolling(window=20).sum()/(np.sqrt(20)*data["sigma"])
    data["profit63"]=data["profit"].rolling(window=63).sum()/(np.sqrt(63)*data["sigma"])
    data["profit126"]=data["profit"].rolling(window=126).sum()/(np.sqrt(126)*data["sigma"])
    data["profit252"]=data["profit"].rolling(window=252).sum()/(np.sqrt(252)*data["sigma"])
    data["close_sigma"]=data["close"].rolling(window=63).std()
    data["macd1"]=(data["close"].ewm(span=8,adjust=False).mean()-data["close"].ewm(span=24,adjust=False).mean())/data["close_sigma"]
    data["macd2"]=(data["close"].ewm(span=16,adjust=False).mean()-data["close"].ewm(span=48,adjust=False).mean())/data["close_sigma"]
    data["macd3"]=(data["close"].ewm(span=32,adjust=False).mean()-data["close"].ewm(span=96,adjust=False).mean())/data["close_sigma"]
    data["profit"] = data["profit"]/data["sigma"]
    data["expect"] = data["profit"].shift(-1)


    data=data[["date","profit","profit20","profit63","profit126","profit252","macd1","macd2","macd3","expect"]]
    # data=pd.DataFrame({"A":[1,2,3,4,5,6,7,8,9],"B":[2,3,1,3,43,1,32,13,4]})
    data=data.dropna()
    print(data)
    print(data.columns)
    data.to_excel(f"strategy/lstm/train_data/{future}.xlsx")
    
typelist = ['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']





a=pd.read_excel("data/backup/CU_daily.xlsx")
# print(a["close"].to_numpy())
# fft= np.fft.fft(a["close"].to_numpy(),252)
# x=pd.DataFrame(np.abs(fft))
# print(x)
# x.loc[1:].plot()
# plt.show()

y = a[["close","volume"]]
print(y)
y["close"]=y['close']/30000
y["volume"]=y["volume"]/y["volume"].mean()
print(y)
y.plot()
plt.show()