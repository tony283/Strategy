import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("strategy/")
import os
import multiprocessing
from utils.BackTestEngine import *
#order_target_num 用来下空单或多单
#sell_target_num 用来平仓
class Section_Momentum_BackTest(BackTest):
    def init(self, context):
        context.R=20
        context.H=20
        context.name=f"section_R{context.R}_H{context.H}"
        context.fired=False
        context.typelist=['AG', 'HC', 'J', 'RB', 'SS', 'FU', 'PP', 'SP', 'EB', 'PG', 'A', 'M', 'OI', 'SR', 'Y', 'B', 'P', 'CU', 'LC', 'SI', 'PX']    
        context.count=0#用于计时
        context.range=0.2#取前20%
        for item in context.typelist:
            self.subscribe(item)#注册品种
        #print(self.data)
    def before_trade(self, context,m_data):
        if context.fired:
            context.count+=1
        
        
    # @timer
    def handle_bar(self, m_data, context):
        if context.fired:
            if context.count<context.H:
                return
            else:
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
            ##开始计算每个品种的收益率
            temp_dict =[]#用于储存收益率信息
            for future_type in context.typelist:
                try:
                    profit = (m_data[future_type]["close"].iloc[-1]-m_data[future_type]["close"].iloc[-1-context.R])/m_data[future_type]["close"].iloc[-1-context.R]
                    temp_dict.append([future_type,profit])
                except:
                    continue
            ranking = pd.DataFrame(temp_dict,columns=["future_type","profit"])
            range=int(context.range*len(ranking))
            ranking = ranking.sort_values(by="profit",ascending=True)#排名
            cash_max = (self.position.cash//(range*2))/10000
            for index, row in ranking.iloc[-range:].iterrows():#收益率最高的
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"long")
            for index, row in ranking.iloc[:range].iterrows():#收益率最低的
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"short")
            context.fired=True

    def after_trade(self, context):
        pass
        
        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for r in range(2,20):
        for h in [1,2,3,4,5,6,7,8]:
            engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.R=r
            engine.context.H=h
            engine.context.name = f"smallsecmm_R{r}_H{h}"
            p.apply_async(engine.loop_process,args=("20120101","20240501","back/smallsec/smallsecmm/"))
            # engine.loop_process(start="20120101",end="20240501",saving_dir="back/section/newsecrange/")
    # print("-----start-----")
    p.close()
    p.join()
    # print("------end------")