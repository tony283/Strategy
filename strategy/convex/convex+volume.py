import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
sys.path.append("C:\\Users\\ROG\\Desktop\\Strategy\\strategy\\utils")
from BackTestEngine import *
import multiprocessing
class Convex(BackTest):
    def init(self, context):
        #context可以自定义变量并用在其他函数
        context.name="Convex"
        context.N=20 #分段n日收益率
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
                    if len(m_data[future_type])<=3+3*context.N+context.M:
                        continue
                    volume = m_data[future_type]["volume"].iloc[-1-context.M-3*context.N:-1-context.M].mean()
                    if volume==0:
                        continue
                    volume0=m_data[future_type]["volume"].iloc[-1-context.M-context.N:-1-context.M].mean()
                    volume1=m_data[future_type]["volume"].iloc[-1-context.M-2*context.N:-1-context.M-context.N].mean()
                    volume2=m_data[future_type]["volume"].iloc[-1-context.M-3*context.N:-1-context.M-2*context.N].mean()
                    if volume0==0 or volume1==0 or volume2==0:
                        continue
                    p0=m_data[future_type]["profit"].iloc[-2-context.M]/volume0
                    p1=m_data[future_type]["profit"].iloc[-2-context.M-context.N]/volume1
                    p2=m_data[future_type]["profit"].iloc[-2-context.M-2*context.N]/volume2
                    convex=p0+p2-2*p1

                    daily_temp_dict.append([future_type,convex])
                except:
                    pass
            daily_ranking = pd.DataFrame(daily_temp_dict,columns=["future_type","convex"])    
            daily_ranking = daily_ranking.sort_values(by="convex",ascending=True)
            Nd=len(daily_ranking)
            m_range = int(context.range*Nd)
            cash_max = (self.position.cash//(m_range))/10000
            # for future_type in daily_ranking["future_type"].iloc[:m_range]:#smooth动量最低的
            #     close = m_data[future_type]["close"].iloc[-1]
            #     multi = m_data[future_type]["multiplier"].iloc[-1]
            #     self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"short")
            for future_type in daily_ranking["future_type"].iloc[-m_range:]:#smooth动量最高的
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"long")
            context.fired=True

        pass
    def after_trade(self, context):
        #收盘后做一些事情
        pass
        
if(__name__=="__main__"):
    for n in [10,20,60,120]:
        for m in [0,1,2,5]:
            engine = Convex(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.N=n
            engine.context.M = m
            engine.context.name=f"convex+volume_N{n}_M{m}"
            engine.loop_process("20120101","20240801")