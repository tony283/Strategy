from typing import Any
import rqdatac
import pandas as pd
#rqdatac.init()
from datetime import datetime
import os
import json
import sys



def transfer_dominant(m_data,future_type, fee=0):
    cache ={}
    base_info = m_data.iloc[0]
    dominant = base_info['dominant']
    base_dominant = pd.read_excel(f"data/{future_type}/{dominant}.xlsx",index_col="date") 
    cache[base_info["dominant"]]= base_dominant
    original_price = base_dominant.loc[base_info["date"],"prev_close"]
    price = original_price
    price_list=[]
    for i in range(len(m_data)):
        base_info = m_data.iloc[i]
        dominant = base_info['dominant']
        if base_info["dominant"] not in cache.keys():
            cache[base_info["dominant"]]= pd.read_excel(f"data/{future_type}/{dominant}.xlsx",index_col="date") 
        price = price*cache[base_info["dominant"]].loc[base_info["date"],"close"]/cache[base_info['dominant']].loc[base_info["date"],"prev_close"]
        if base_info["switch"]==1:
            price*=(1-fee)
        price_list.append(price)
    m_data["real_price"]=price_list
    m_data:pd.DataFrame = m_data
    m_data.to_excel(f"data/{future_type}/{future_type}_dominant_price.xlsx",index=False)
            
       

