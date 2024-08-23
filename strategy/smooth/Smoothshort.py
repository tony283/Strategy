import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
sys.path.append("C:\\Users\\ROG\\Desktop\\Strategy\\strategy\\utils")
from BackTestEngine import *
import multiprocessing
class Smooth(BackTest):
    def init(self, context):
        #context可以自定义变量并用在其他函数
        context.name="Smooth"
        context.N=20
        context.M=0
        context.fired=False
        context.typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
        for item in context.typelist:
            self.subscribe(item)#注册品种
        context.count =0 #用于计数
        context.day =20#调仓周期
        context.range=0.2
        
            
        
        
        
    def before_trade(self, context):
        if context.fired:
            context.count += 1
        #开盘前做一些事
        pass
    def handle_bar(self, m_data, context):
        if context.fired:
            if context.count<context.day:
                return
            else:
                #平仓
                for future_type_dir, amount in self.position.hold.items():
                    info = future_type_dir.split("_")
                    future_type = info[0]
                    direction = info[1]
                    multi = m_data[future_type]["multiplier"].iloc[-1]
                    if(amount[0]//multi<=0):
                        continue
                    self.sell_target_num(m_data[future_type]["close"].iloc[-1],amount[0]//multi,multi,future_type,direction)
                    context.count=0
                    context.fired=False
            
            
                    
        if not context.fired:
            daily_temp_dict =[]
            for future_type in context.typelist:
                
                try:
                    if len(m_data[future_type])<=2+context.N+context.M:
                        continue
                    profit_total = m_data[future_type]["profit"].iloc[-2-context.N-context.M:-1-context.M].apply(lambda  x :abs(x)).sum()
                    profit = (m_data[future_type]["close"].iloc[-2-context.M]-m_data[future_type]["close"].iloc[-2-context.N-context.M])/m_data[future_type]["close"].iloc[-2-context.N-context.M]
                    if(profit_total==0):
                        continue
                    
                    smooth = profit/profit_total
                    daily_temp_dict.append([future_type,smooth])
                except:
                    pass
            daily_ranking = pd.DataFrame(daily_temp_dict,columns=["future_type","smooth"])    
            daily_ranking = daily_ranking.sort_values(by="smooth",ascending=True)
            Nd=len(daily_ranking)
            
            m_range = int(context.range*Nd)
            cash_max = (self.position.cash//(m_range))/10000
            for future_type in daily_ranking["future_type"].iloc[:m_range]:#smooth动量最低的
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"short")
            # for future_type in daily_ranking["future_type"].iloc[-m_range:]:#smooth动量最高的
            #     close = m_data[future_type]["close"].iloc[-1]
            #     multi = m_data[future_type]["multiplier"].iloc[-1]
            #     self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"long")
            context.fired=True

        pass
    def after_trade(self, context):
        #收盘后做一些事情
        pass
        
if(__name__=="__main__"):
    for n in [20,60,120,180,240]:
        for m in [0,1,5]:
            engine = Smooth(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.N=n
            engine.context.M = m
            engine.context.name=f"smoothshort_N{n}_M{m}"
            engine.loop_process("20120101","20240801")