import pandas as pd
import numpy as np
from strategy.utils.utils import *
from datetime import datetime
from dateutil import rrule
import matplotlib.pyplot as plt
import random
# # typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
# #           'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
# #           'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
# #           'SI', 'SH', 'PX', 'BR', 'AO']
# # for t in typelist:
# #     future_data = pd.read_excel("data/"+t+"_daily.xlsx")
# #     future_data["profit"]=(future_data["close"]-future_data["prev_close"])/future_data["prev_close"]
    
# #     for sig in [5,20,40,252,126,63]:
# #         future_data["sigma"+str(sig)]=future_data['profit'].rolling(window=sig,min_periods=1).std().shift(fill_value=0)
# #     future_data.to_excel("data/"+t+"_daily.xlsx")
        
# # future_data = pd.read_excel("data/"+"AU"+"_daily.xlsx")        
# # print(future_data[["close","sigma20","sigma40","sigma63","sigma126","sigma252"]].corr() )
# N=2000
# r=0.1       
# test =[pow((1+r/252),i) for i in range(N)]
# test= np.array(test)
# profit=np.diff(test)/test[:-1]
# print(profit)
# sigma = np.sqrt(252)*profit.std()
# profit_annual = np.log(test[-1]/test[0])/(N/252)
# print(f"sigma is {sigma},profit is {profit_annual},division is {profit_annual/sigma}")

# date =datetime(2015,1,1)
newlist = ['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'BU', 'FG',  'L', 'MA', 'PP', 'RU',
           'TA', 'V', 'A', 'C', 'CF', 'M', 'OI', 'RM', 'SR', 'Y', 'JD',  'B', 'P', 'AL', 'CU', 'PB', 'ZN']
# print(len(newlist))
# # # for i in datalist:
# # #     a =pd.read_excel(f"data/{i}_daily.xlsx")
# # #     if(date>a.iloc[0]["date"]):
# # #         newlist.append(i)
# #############################################################################
newlist_c = newlist.copy()
dates = pd.read_excel("data/trading_dates.xlsx")
dates = dates[dates["date"]>=datetime(2015,1,1)]
dates = dates[dates["date"]<=datetime(2014,8,8)]
df = pd.DataFrame(columns=["R","profitall63","sigma5","sigma20","sigma63","vol","open_interest",*[f"profit{i}" for i in range(1,20)]])

# print(df)


for i in newlist:
    df_temp = pd.DataFrame(columns=["R","profitall63","sigma5","sigma20","sigma63","vol","open_interest","profit1","profit2","profit3","profit4","profit5"])
    future_his = pd.read_excel(f"data/{i}_daily.xlsx")
    future_his["R"] = future_his["profit"].shift(-1)
    future_his["profit63"]=(future_his["close"].shift()-future_his["close"].shift(64))/future_his["close"].shift(64)
    future_his["profit1"]=(future_his["close"].shift()-future_his["close"].shift(2))/future_his["close"].shift(2)
    future_his["profit2"]=(future_his["close"].shift(2)-future_his["close"].shift(3))/future_his["close"].shift(3)
    future_his["profit3"]=(future_his["close"].shift(3)-future_his["close"].shift(4))/future_his["close"].shift(4)
    future_his["profit4"]=(future_his["close"].shift(4)-future_his["close"].shift(5))/future_his["close"].shift(5)
    future_his["profit5"]=(future_his["close"].shift(5)-future_his["close"].shift(6))/future_his["close"].shift(6)
    for i in range(1,20):
        future_his[f"profit{i}"]=(future_his["close"].shift(i)-future_his["close"].shift(i+1))*10/future_his["close"].shift(2)
        

    
    future_his =future_his.set_index("date")
    df_temp["profitall63"] = future_his["profit63"]*10
    df_temp["sigma5"] = future_his["sigma5"]*10
    df_temp["sigma20"] = future_his["sigma20"]*10
    df_temp["sigma63"] = future_his["sigma63"]*10
    for i in range(1,20):
        df_temp[f"profit{i}"] = future_his[f"profit{i}"]*10

    df_temp["vol"]=(future_his["volume"].shift()/future_his["volume"].shift(2)).apply(lambda x :np.tanh(x-1))
    df_temp["open_interest"] = (future_his["open_interest"].shift()/future_his["open_interest"].shift(2)).apply(lambda x :np.tanh(x-1))
    df_temp["R"] = future_his["R"]*10
    df = pd.concat([df,df_temp])
df = df.dropna(axis=0,how="any")
df.to_excel("data/DQN/onedata.xlsx")

# df = pd.read_excel("data/DQN/onedata.xlsx")
# print(df.isnull().any())
# print(df)
# df = pd.read_excel("data/DQN/alldata.xlsx")
# df =df[newlist]
# a=df.iloc[0].tolist()
# print(a)
