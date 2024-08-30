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
print(len(newlist))
# # for i in datalist:
# #     a =pd.read_excel(f"data/{i}_daily.xlsx")
# #     if(date>a.iloc[0]["date"]):
# #         newlist.append(i)
# newlist_c = newlist.copy()
# newlist_c.extend([i+"Current" for i in  newlist])
# newlist_c.extend([i+"Reward" for i in  newlist])
# dates = pd.read_excel("data/trading_dates.xlsx")
# dates = dates[dates["date"]>=datetime(2015,1,1)]
# dates = dates[dates["date"]<=datetime(2014,8,8)]
# df = pd.DataFrame(index=dates['date'].to_list(),columns=["date",*newlist_c])

# print(df)


# for i in newlist:
#     future_his = pd.read_excel(f"data/{i}_daily.xlsx")
#     future_his =future_his[["date","close","prev_close","profit","sigma5","sigma63"]]
#     future_his["R"] = future_his["profit"].shift(-1)
#     future_his["profit63"]=(future_his["close"].shift()-future_his["close"].shift(64))/future_his["close"].shift(64)
#     future_his["p/v1"]=future_his["profit"].shift()/future_his["sigma5"]
#     future_his["p/v63"]=future_his["profit63"]/(future_his["sigma63"]*7.937)
#     future_his =future_his[future_his["date"]>=datetime(2015,1,1)].set_index("date")
#     print(future_his)
#     df[i] =future_his["p/v63"]
#     df[i+"Current"] = future_his["p/v1"]
#     df[i+"Reward"] = future_his["R"]
#     print(df)
# # df.to_excel("data/DQN/alldata.xlsx")

# df = pd.read_excel("data/DQN/alldata.xlsx")

# print(df)
# df = pd.read_excel("data/DQN/alldata.xlsx")
# df =df[newlist]
# a=df.iloc[0].tolist()
# print(a)
import torch
a=np.array([12,3,4,5])
b=np.array([1,1,2,3])
a=torch.tensor(a).unsqueeze(0)
b=torch.tensor(b).unsqueeze(0)
c=torch.tensor([[1,2,3]],  dtype=torch.long)
print(float(a[0]@b[0]))