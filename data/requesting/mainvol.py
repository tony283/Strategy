import requestdata
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

R=17
H=2
PROFIT=1.2

try:
    REALCASH = int(sys.argv[1])
except:
    
    REALCASH = 100000
USINGRATE=0.5

CASH=REALCASH*10*USINGRATE



BookTable = pd.DataFrame(columns=["future_type","total_turnover","volume","direction","price"])
typelist = ['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
# requestdata.request_all()
m_data = requestdata.request_old_data()
current = m_data["CU"]["date"].iloc[-1]
print(str(current)[:10])
multiplier = pd.read_excel("C:/Users/ROG/Desktop/Strategy/data/multiplier.xlsx",index_col=0)
temp_dict =[]#用于储存收益率信息

for future_type in typelist:
    try:
        profit_max = m_data[future_type].iloc[-R:].copy()
        profit_max["abs_profit"]= profit_max["profit"].apply(lambda x : abs(x))
        
        close_mean = profit_max["close"].mean()
        profit_max = profit_max["abs_profit"].max()
        if profit_max==0 or profit_max>PROFIT:
            continue
        close =m_data[future_type]["close"].iloc[-1]
        direction = "long" if close<=close_mean else "short"
        temp_dict.append([future_type,direction])
        
    except:
        continue
    
ranking = pd.DataFrame(temp_dict,columns=["future_type","direction"])
if len(ranking)!=0:
    cash_max = (CASH/(len(ranking)))
    
    for index, row in ranking.iterrows():
        future_type=row["future_type"]
        close = m_data[future_type]["close"].iloc[-1]
        multi = multiplier.loc[future_type,"multiplier"]
        
        buy_amount = int(cash_max/(close*multi))
        if buy_amount<=0:
            continue
        BookTable.loc[len(BookTable)]=[future_type,buy_amount*close*multi,int(cash_max/(close*multi)),row["direction"],close]
else:
    pass


print(BookTable)
# BookTable.to_excel(f"data/requesting/{str(current)[:10]}.xlsx")