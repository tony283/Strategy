import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#sys.path.append("c:\\Users\\ROG\\Desktop\\Strategy\\utils")
import os

from utils.BackTestEngine import *
#order_target_num 用来下空单或多单
#sell_target_num 用来平仓
class Section_Momentum_BackTest(BackTest):
    def init(self, context):
        context.R=5
        context.H1=20
        context.H2=15
        context.name="best_section"
        context.h1fired=False
        context.h2fired=False
        context.above=[]
        context.below=[]
        context.typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
        context.count1=0#用于计时
        context.count2=0#用于计时
        context.range=0.2#取前20%
        for item in context.typelist:
            self.subscribe(item)#注册品种
        #print(self.data)
    def before_trade(self, context):
        if context.h1fired:
            context.count1+=1
        if context.h2fired:
            context.count2+=1
        
        
        return super().before_trade(context)
    
    def handle_bar(self, m_data, context):
        if context.h1fired:
            if context.count1<context.H1:
                return
            else:
                for future_type in context.above:
                    direction = "long"
                    multi = m_data[future_type]["multiplier"].iloc[-1]
                    amount = self.position.hold[future_type+"_long"]
                    if(amount[0]//multi<=0):
                        continue
                    self.sell_target_num(m_data[future_type]["close"].iloc[-1],amount[0]//multi,multi,future_type,direction)
                    context.count1=0
                    context.above.remove(future_type)
                    context.h1fired=False
        if context.h2fired:
            if context.count2<context.H2:
                return
            else:
                for future_type in context.below:
                    direction = "long"
                    multi = m_data[future_type]["multiplier"].iloc[-1]
                    amount = self.position.hold[future_type+"_long"]
                    if(amount[0]//multi<=0):
                        continue
                    self.sell_target_num(m_data[future_type]["close"].iloc[-1],amount[0]//multi,multi,future_type,direction)
                    context.count2=0
                    context.below.remove(future_type)
                    context.h2fired=False
        if not context.h1fired:
            ##开始计算每个品种的收益率
            temp_dict =[]#用于储存收益率信息
            for future_type in context.typelist:
                try:
                    profit = (m_data[future_type]["close"].iloc[-2]-m_data[future_type]["close"].iloc[-2-context.R])/m_data[future_type]["close"].iloc[-2-context.R]
                    temp_dict.append([future_type,profit])
                except:
                    continue
            ranking = pd.DataFrame(temp_dict,columns=["future_type","profit"])
            range=int(self.context.range*len(ranking))
            ranking = ranking.sort_values(by="profit",ascending=True)#排名
            cash_max = (self.position.cash//(range*(2 if not context.h1fired and not context.h2fired else 1)))/10000
            for index, row in ranking.iloc[-range:].iterrows():#收益率最高的
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"long")
                context.above.append(future_type)
            
            context.h1fired=True
        if not context.h2fired:
            ##开始计算每个品种的收益率
            temp_dict =[]#用于储存收益率信息
            for future_type in context.typelist:
                try:
                    profit = (m_data[future_type]["close"].iloc[-2]-m_data[future_type]["close"].iloc[-2-context.R])/m_data[future_type]["close"].iloc[-2-context.R]
                    temp_dict.append([future_type,profit])
                except:
                    continue
            ranking = pd.DataFrame(temp_dict,columns=["future_type","profit"])
            range=int(self.context.range*len(ranking))
            ranking = ranking.sort_values(by="profit",ascending=True)#排名
            cash_max = (self.position.cash//(range))/10000
            for index, row in ranking.iloc[:range].iterrows():#收益率最低的
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"long")
                context.below.append(future_type)
            context.h2fired=True

    def after_trade(self, context):
        pass
        
        



engine = Section_Momentum_BackTest(cash=100000000,margin_rate=1,margin_limit=0,debug=True)
engine.loop_process(start="20150101",end="20240501")