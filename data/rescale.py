from typing import Any
import rqdatac
import pandas as pd
rqdatac.init()
from datetime import datetime
import os
import json
import sys
def timer(func):
    def func_wrapper(*args,**kwargs):
        from time import time
        time_start = time()
        result = func(*args,**kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper
def str2list(s:str):
    s=s.replace("[","")
    s=s.replace("]","")
    s=s.replace("'","")
    l = s.split(",")
    l =[i.strip() for i in l]
    return l

def transfer_dominant(m_data,future_type, fee=0):
    cache ={}
    base_info = m_data.iloc[0]
    print(base_info)
    base_dominant = pd.read_excel(f"data/{future_type}/{base_info["dominant"]}.xlsx",index_col="date") 
    cache[base_info["dominant"]]= base_dominant
    original_price = base_dominant.loc[base_info["date"],"prev_close"]
    price = original_price
    price_list=[]
    for i in range(len(m_data)):
        base_info = m_data.iloc[i]
        if base_info["dominant"] not in cache.keys():
            cache[base_info["dominant"]]= pd.read_excel(f"data/{base_info["dominant"][:-4]}/{base_info["dominant"]}.xlsx",index_col="date") 
        price = price*cache[base_info["dominant"]].loc[base_info["date"],"close"]/cache[base_info["dominant"]].loc[base_info["date"],"prev_close"]
        if base_info["switch"]==1:
            price*=(1-fee)
        price_list.append(price)
    m_data["real_price"]=price_list
    m_data:pd.DataFrame = m_data
    m_data.to_excel(f"data/{base_info["dominant"][:-4]}/{base_info["dominant"]}.xlsx",index=False)
            
       

future_type = "CU"
m_data =pd.read_excel(f"data/{future_type}/{future_type}_dominant_result.xlsx")
print(m_data.iloc[0])
#transfer_dominant(m_data,future_type)
