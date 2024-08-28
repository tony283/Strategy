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
N=2000
r=0.1       
test =[pow((1+r/252),i) for i in range(N)]
test= np.array(test)
profit=np.diff(test)/test[:-1]
print(profit)
sigma = np.sqrt(252)*profit.std()
profit_annual = np.log(test[-1]/test[0])/(N/252)
print(f"sigma is {sigma},profit is {profit_annual},division is {profit_annual/sigma}")
print(len(['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']))