import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
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
        
engine = Sample(cash=100000000,margin_rate=0.2,margin_limit=0.8)#可以设置保证金比例和补交保证金的阈值
engine.subscribe("CU")#可以订阅更多品种，如果是多品种可以在init函数中循环self.subscribe(future_type)
engine.loop_process(start="20130101",end="20240501")#回测时间区间
