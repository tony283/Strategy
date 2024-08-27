import pandas as pd
import numpy as np
from strategy.utils.utils import *
from datetime import datetime
import datetime
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
        
# # print(a[a["A"]==5])
# # a:pd.DataFrame = pd.read_excel("data/CU_daily.xlsx")
# # print(a.dtypes)
# # print(a[a["date"]>=datetime.datetime.strptime("20100109","%Y%m%d")].index[0])
# # print(a)
# # print(a[a["date"]<datetime.datetime.strptime("20120109","%Y%m%d")][a["date"]>datetime.datetime.strptime("20110509","%Y%m%d")][:1])
# # a=a._append(a[a["date"]<datetime.datetime.strptime("20120109","%Y%m%d")][a["date"]>datetime.datetime.strptime("20110509","%Y%m%d")][:1])
# # print(a)
# # a["close"]=a["close"].apply(lambda a:a*10000)
# # print(a)
# # a="CU_short"
# # b=a.split("_")
# # print(b)
# # date1=datetime.datetime(1991,1,2)
# # date2=datetime.datetime(1992,1,1)
# # print((date2-date1).days)
# # a=[1,2,3]
# # b=[0]
# # c=b.extend(a)
# # print(b)
# # test = ["a","b","c","d","e"]
# # remove = ["b","d"]
# # for i in test:
# #     if i in remove:
# #         test.remove(i)
        
# # print(test)

# # test={"1":lambda x:x+1,"2":lambda x:x*x}
# # # print(test["2"](10))
# # import pandas as pd

# # # 创建示例数据框
# # data = {'x': [1, 2, 3, 4, 5], 'y': [5, 4, 3, 2, 1]}
# # df = pd.DataFrame(data)

# # # 创建rolling对象并计算滚动相关系数
# # rolling_corr = df['x'].rolling(window=3,min_periods=1).mean()
# # print(rolling_corr)
# # vol_bottom_data= pd.read_excel("data/factors/Back_volbottomtargetvol_S20_T0.25_day62.xlsx")
# # vol_top_data=pd.read_excel("data/factors/Back_voltoptargetvol_S5_T0.15_day62.xlsx")
# # for i in [20,40,63,126,252]:
# #     vol_bottom_data["profit"+str(i)]=(vol_bottom_data["volbottomtargetvol_S20_T0.25_day62"].shift(1)-vol_bottom_data["volbottomtargetvol_S20_T0.25_day62"].shift(1+i))/vol_bottom_data["volbottomtargetvol_S20_T0.25_day62"].shift(1+i)
# #     vol_bottom_data["profit"+str(i)] = vol_bottom_data["profit"+str(i)].fillna(0)
    
# #     vol_top_data["profit"+str(i)]=(vol_top_data["voltoptargetvol_S5_T0.15_day62"].shift(1)-vol_top_data["voltoptargetvol_S5_T0.15_day62"].shift(1+i))/vol_top_data["voltoptargetvol_S5_T0.15_day62"].shift(1+i)
# #     vol_top_data["profit"+str(i)]=vol_top_data["profit"+str(i)].fillna(0)

# # print(vol_bottom_data)
# # print(vol_top_data)
# # vol_bottom_data.to_excel("data/factors/Back_volbottomtargetvol_S20_T0.25_day62.xlsx")
# # vol_top_data.to_excel("data/factors/Back_voltoptargetvol_S5_T0.15_day62.xlsx")
typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']    
# turnover = pd.DataFrame(columns=["turnover"],index=typelist
#                         )
# for i in typelist:
    
#     turnover.loc[i,"turnover"]= random.randint(1,1000)
    
    
# print(turnover)
# print(turnover.loc[2:])
# #turnover.to_excel("Report/barrier/future_turnover.xlsx")
    
# # a=pd.DataFrame({"A":[1,23,4],"B":[4,2,1],"C":[64,2,4]})

# a=np.array([0,1,2,3,3])
# print(a.std())
# # print(a["B"].index)


# a=0.1234544123
# print(f"{a:.2f}")

# 定义迷宫的地图和起点、终点
# a = pd.read_excel(r"C:\Users\ROG\Desktop\Strategy\back\trade\Tradebarrierlong_M0.150.xlsx")
# a=a.groupby(by="type").size().reset_index(name="count")
# a.to_excel("Report/barrier/future_count.xlsx")
import sys
import time
for i in range(1, 101):
    print("\r", end="")
    print(f"{i} ", end="")
    sys.stdout.flush()
    time.sleep(0.5)