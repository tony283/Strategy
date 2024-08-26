import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
sys.path.append("C:\\Users\\ROG\\Desktop\\Strategy\\strategy\\utils")
from BackTestEngine import *
import multiprocessing
class Barrier(BackTest):
    def init(self, context):
        #context可以自定义变量并用在其他函数
        context.name="Convex"
        #context.N=20 #排名依据
        context.M=0.95
        context.fired=False
        context.typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
        for item in context.typelist:
            self.subscribe(item)#注册品种
        context.count =0 #用于计数
        context.day =20#调仓周期
        context.range=0.15
        context.storage = pd.DataFrame(columns=context.typelist,index=["factor"])
        for i in context.typelist:
            context.storage.loc["factor",i]=0

        
        
    def before_trade(self, context):
        if context.fired:
            context.count += 1
        #开盘前做一些事
        pass
    def handle_bar(self, m_data, context):
        for i in context.typelist:
            try:
                temp = m_data[i].iloc[-2]
                
                if(temp["volume"]*temp["multiplier"]*(temp["high"]+temp["low"]+temp["close"]+temp["open"])*0.25==0):
                    continue
                # m_mean=m_data[i].iloc[max(len(m_data[i])-41,0):-1]
                # m_mean_c =m_mean.copy()
                # m_mean_c["turnover"]=m_mean_c["volume"]*m_mean_c["multiplier"]*(m_mean_c["high"]+m_mean_c["low"]+m_mean_c["close"]+m_mean_c["open"])*0.25
                # m_mean=m_mean_c
                #turnover = m_mean["turnover"].mean()
                
                context.storage.loc["factor",i]=context.storage[i].loc["factor"]*context.M  +  temp["profit"]*10000000/(temp["volume"]*temp["multiplier"]*(temp["high"]+temp["low"]+temp["close"]+temp["open"])*0.25)
            except:
                continue
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
            daily_ranking :pd.DataFrame = context.storage.T
            daily_ranking = daily_ranking[daily_ranking["factor"]!=np.nan]
            daily_ranking = daily_ranking[daily_ranking["factor"]!=0]
            daily_ranking = daily_ranking.sort_values(by="factor",ascending=False)
            Nd=len(daily_ranking)
            m_range = int(context.range*Nd)
            cash_max = (self.position.cash//(m_range))/10000
            # for future_type in daily_ranking["future_type"].iloc[:m_range]:#smooth动量最低的
            #     close = m_data[future_type]["close"].iloc[-1]
            #     multi = m_data[future_type]["multiplier"].iloc[-1]
            #     self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"short")
            for future_type in daily_ranking.index[:m_range]:#smooth动量最高的
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
    for m in [i*0.025 for i in range(40)]:
        engine = Barrier(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
        engine.context.M = m
        engine.context.name=f"barrierlong_M{m:.3f}"
        p.apply_async(engine.loop_process,args=("20120101","20240801"))
    print("-----start-----")
    p.close()
    p.join()
    print("------end------")
    # for m in [i*0.025 for i in range(40)]:
    #     engine = Barrier(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
    #     engine.context.M = m
    #     engine.context.name=f"barrierlongstd_M{m:.3f}"
    #     engine.loop_process("20180101","20240801")