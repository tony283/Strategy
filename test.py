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
class sc:
    def __init__(self) -> None:
        self.dic2={}
    def async_read(self,future_type):
        
        self.dic2[future_type]=pd.read_excel("data/"+future_type+"_daily.xlsx")
    def get_result(self,future):
        result=future.result()
        self.dic2[result[0]]=result[1]
# 7059
# 过去三年存续最大值
# 流水
# 起息日到期日
# 期限大于半年为租赁
if __name__=="__main__":
    typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
                'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
                'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
                'SI', 'SH', 'PX', 'BR', 'AO']
        
    time1=time.time()
    dic1= dict()
    dic2= dict()
    # for i in typelist:
    #     dic1[i] = pd.read_excel("data/"+i+"_daily.xlsx")
    time2=time.time()
    print(f"ord{time2-time1}")
    temp=sc()
    with ProcessPoolExecutor(20) as executor: # 创建 ThreadPoolExecutor 
        future_list = executor.map(temp.async_read, [file for file in typelist]) # 提交任务

    
    for future in future_list:
        
        pass

    print(time.time()-time2)
    print(temp.dic2["CU"])