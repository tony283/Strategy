import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("strategy/")
import os
import multiprocessing
from utils.BackTestEngine import *

class Sample(BackTest):
    def init(self, context):
        #context可以自定义变量并用在其他函数
        context.name="momentum"
        context.fired=False
    def before_trade(self, context):
        #开盘前做一些事
        pass
    def handle_bar(self, m_data, context):
        #m_data存储了订阅品种的直到本交易日的所有数据
        #order_target_num方法用来下空单或多单
        #sell_target_num方法 用来平仓
        pass
    def after_trade(self, context):
        #收盘后做一些事情
        pass
"""
双参数网格扫描，两个参数列表填在for循环中，
网格扫描的结果可以用back/merge_plot_section2.py对结果进行分析。
"""        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for n in [20]:
        for h in range(1,2):
            engine = Sample(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.H=h
            engine.context.R=n
            #name必须要有，这个是每个回测曲线的id，也是生成结果的名字
            engine.context.name = f"newsectest_R{n}_H{h}"
            # p.apply_async(engine.loop_process,args=("20180101","20241030","back/section/newsecXGB/"))
            engine.loop_process(start="20180101",end="20241030",saving_dir="back/section/newsecXGB/")
    # print("-----start-----")
    p.close()
    p.join()