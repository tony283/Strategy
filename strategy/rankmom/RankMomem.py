import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
sys.path.append("C:\\Users\\ROG\\Desktop\\Strategy\\strategy\\utils")
from BackTestEngine import *
import multiprocessing
class Rank(BackTest):
    def init(self, context):
        #context可以自定义变量并用在其他函数
        context.name="Rank"
        context.N=120
        context.M=20
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
        
    
        
        context.rank_factor = pd.DataFrame(columns=context.typelist)
        
            
        
        
        
    def before_trade(self, context,m_data):
        if context.fired:
            context.count += 1
        #开盘前做一些事
        pass
    def handle_bar(self, m_data, context):
        #每日排名
        daily_temp_dict =[]
        context.rank_factor.loc[len(context.rank_factor)] = [float(0) for i in context.typelist]
        for future_type in context.typelist:
            
            try:
                profit = m_data[future_type]["profit"].iloc[-2]
                daily_temp_dict.append([future_type,profit])
            except:
                pass
        daily_ranking = pd.DataFrame(daily_temp_dict,columns=["future_type","profit"])    
        daily_ranking = daily_ranking.sort_values(by="profit",ascending=True)
        Nd=len(daily_ranking)
        temp = [(i-(Nd+1)/2)/np.sqrt((Nd*Nd-1)/12) for i in range(1,Nd+1)]
        
        daily_ranking["Rank"] = temp
        for index in daily_ranking.index:
            context.rank_factor.loc[len(context.rank_factor)-1, daily_ranking.loc[index,"future_type"]]=daily_ranking.loc[index,"Rank"]
        #print(context.rank_factor)
        #排名结束
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
            if(len(context.rank_factor)-context.N-context.M-1<0):
                return
            ranking_list = context.rank_factor.iloc[-context.N-context.M-1:-context.M-1].mean(axis=0).sort_values(ascending=False)#.index.tolist()
            ranking_list =ranking_list[ranking_list!=0]
            ranking_list = ranking_list.index.tolist()
            m_range = int(context.range*len(ranking_list))
            cash_max = (self.position.cash//(m_range))/10000
            for future_type in ranking_list[:m_range]:#rank动量最高的
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"long")#自动切换
            context.fired=True
            
            
            
        #m_data存储了订阅品种的直到本交易日的所有数据
        #order_target_num方法用来下空单或多单
        #sell_target_num方法 用来平仓
        pass
    def after_trade(self, context):
        #收盘后做一些事情
        pass
        
if(__name__=="__main__"):
    for n in [20,60,120,180,240]:
        for m in [0,20,60]:
            engine = Rank(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.N=n
            engine.context.M = m
            engine.context.name=f"rank_N{n}_M{m}"
            engine.loop_process("20120101","20240801")