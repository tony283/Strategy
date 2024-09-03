import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#sys.path.append("c:\\Users\\ROG\\Desktop\\Strategy\\utils")
import os

from utils.BackTestEngine import *
#order_target_num 用来下空单或多单
#sell_target_num 用来平仓
class Momentum_BackTest(BackTest):
    def init(self, context):
        context.name="momentum"
        context.day=60
        context.fired=False
    def before_trade(self, context,m_data):
        
        context.day+=1
        
        return super().before_trade(context)
    def handle_bar(self, m_data, context):
        date=m_data["CU"]["date"].iloc[-1]
        
        close=int(m_data["CU"]["close"].iloc[-1]*10000)
        print(f"{date}cash is {self.position.cash},asset is {self.position.asset[-1]},price is {close/10000}")
        margin_rate=self.instrument["margin_rate"]
        multi = m_data["CU"]["multiplier"].iloc[-1]
        if(context.day>60 and not context.fired):
            asset=self.position.asset[-1]
            self.order_target_num(close/10000,int(max(0,self.position.cash-asset*0.5))*int(100/margin_rate)//(100*close*multi),multi,"CU","long")
            

            context.day=0
            context.fired=True
        if(context.day>100 and context.fired):
            self.sell_target_num(close/10000,self.position.hold["CU_long"][0]//multi,multi,"CU","long")
            context.day=60
            context.fired=False
            print(self.position.cash)
        pass
    def after_trade(self, context):
        pass
        
        


engine = Momentum_BackTest(cash=100000000,margin_rate=0.2,margin_limit=0.8)
engine.subscribe("CU")#可以订阅更多品种
engine.loop_process(start="20130101",end="20240501")
