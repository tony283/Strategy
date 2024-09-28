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
        context.fired=False
        context.typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
        for item in context.typelist:
            self.subscribe(item)#注册品种
        context.count =0 #用于计数
        context.H =2#调仓周期
        context.range=0.2
        
            
        
        
        
    def before_trade(self, context,m_data):
        if context.fired:
            context.count += 1
        #开盘前做一些事
        pass
    def handle_bar(self, m_data, context):
        if context.fired:
            if context.count<context.H:
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
                    profit_total = m_data[future_type]["profit"].iloc[-context.N:].apply(lambda  x :abs(x)).sum()
                    profit = m_data[future_type]["profit"].iloc[-context.N:].sum()
                    sigma = m_data[future_type]["profit"].iloc[-20:].std()
                    if(profit_total==0 or sigma==0):
                        continue
                    
                    smooth = profit*abs(profit)/(profit_total*sigma)
                    daily_temp_dict.append([future_type,smooth])
                except:
                    pass
            daily_ranking = pd.DataFrame(daily_temp_dict,columns=["future_type","smooth"])    
            daily_ranking = daily_ranking.sort_values(by="smooth",ascending=True)
            Nd=len(daily_ranking)
            
            m_range = int(context.range*Nd)
            cash_max = (self.position.cash//(2*m_range))/10000
            for future_type in daily_ranking["future_type"].iloc[:m_range]:#smooth动量最低的
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"short")
                context.fired=True

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
    p=multiprocessing.Pool(40)
    for n in [0.1,0.15,0.2,0.25,0.3]:
        for m in [1,2,3,4,5]:
            engine = Smooth(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.range=n
            engine.context.N=15
            engine.context.H=m
            engine.context.name = f"newsecsmoothbreakRange{n}_H{m}"
            p.apply_async(engine.loop_process,args=("20120101","20240501","back/section/newsecsmoothbreak/"))
            # engine.loop_process(start="20120101",end="20240501",saving_dir="back/section/newsecsmooth/")
    # print("-----start-----")
    p.close()
    p.join()
    # print("------end------")