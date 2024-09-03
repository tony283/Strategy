import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#sys.path.append("c:\\Users\\ROG\\Desktop\\Strategy\\utils")
import os

from utils.BackTestEngine import *
#order_target_num 用来下空单或多单
#sell_target_num 用来平仓
class Vol_BackTest(BackTest):
    def init(self, context):
        context.fired=False
        context.vol_pre={"vol":[],"pre":[]}
        pass
    def before_trade(self, context,m_data):
        if context.fired:
            context.days+=1
        
        pass

        
        return super().before_trade(context)
    def handle_bar(self, m_data:pd.DataFrame, context):
        pre_profit19 =(m_data["close"].iloc[-2]-m_data["close"].iloc[-21])/m_data["close"].iloc[-21]
        context.vol_pre["pre"].append(pre_profit19)
        std_vol=m_data["close"].iloc[-6:-1].std()/m_data["close"].iloc[-6:-1].mean()
        context.vol_pre["vol"].append(std_vol)
        
        
        date=m_data["CU"]["date"].iloc[-1]
        asset=self.position.asset[-1]
        close=int(m_data["CU"]["close"].iloc[-1]*10000)
        print(f"{date}cash is {self.position.cash},asset is {self.position.asset[-1]},price is {close/10000}")
        margin_rate=self.instrument["margin_rate"]
        multi = m_data["CU"]["multiplier"].iloc[-1]
        if context.fired:
            if context.days<16:
                return
            self.sell_target_num(close/10000,self.position.hold["CU_long"][0]//multi,multi,"CU","long")
            context.fired=False
            return
        if True:
            self.order_target_num(close/10000,int(max(0,self.position.cash-asset*0.5))*int(100/margin_rate)//(100*close*multi),multi,"CU","long")
        pass
    def after_trade(self, context):
        pass
        
    def factor_builder(self, m_data,context):
        drift=0       
        back=8
        m_data["std_vol"]=((m_data["close"]-m_data["prev_close"])/m_data["prev_close"]).rolling(window=252).std().shift()
        self.drift=drift
        m_data["abs_profit"]=abs((m_data["close"]-m_data["prev_close"])/m_data["prev_close"]).shift(-1)
        for i in [5,20,40,63,126,252]:
            m_data["ave_pre_profit"+str(i)]=(m_data["close"].shift(1)-m_data["close"].shift(1+i))/(m_data["close"].shift(1+i)*i)
        #m_data["vol"+str(context.back)]
        for i in [5,20,40,63,126,252]:
            m_data["ave_profit"+str(i)]=(m_data["close"].shift(-i)-m_data["close"])/(m_data["close"]*i)
        return m_data[["abs_profit","std_vol",*["ave_pre_profit"+str(i) for i in [5,20,40,63,126,252]],*["ave_profit"+str(i) for i in [5,20,40,63,126,252]]]]
    def factor_statistics(self,type):
        m_data=self.data[type].copy()
        m_data:pd.DataFrame=self.factor_builder(m_data,self.context)
        print(m_data)
        ##相关性测试
        corre = m_data.corr()
        corre.to_excel(f"long_term_ave_factor_std_vol_pre_profit_correlation.xlsx")


engine = Vol_BackTest(cash=100000000,margin_rate=0.2,margin_limit=0.8)
engine.subscribe("CU")#可以订阅更多品种
engine.factor_statistics("CU")
