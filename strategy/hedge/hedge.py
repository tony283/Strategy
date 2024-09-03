
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("strategy")
import os
hedge_choose = [("JM","J",0.83422),("RB","HC",0.92982),("SC","LU",0.87724),("L","PP",0.86452),
                ("RU","NR",0.915),("M","RM",0.87272),("Y","OI",0.80065),("P","Y",0.8897),("PX","TA",0.94902),("PF","PX",0.85567)]
from utils.BackTestEngine import *
#order_target_num 用来下空单或多单
#sell_target_num 用来平仓
class Section_Boost_BackTest(BackTest):
    def init(self, context):
        context.hedge_choice = [("JM","J",0.83422),("RB","HC",0.92982),("SC","LU",0.87724),("L","PP",0.86452),
                ("RU","NR",0.915),("M","RM",0.87272),("Y","OI",0.80065),("P","Y",0.8897),("PX","TA",0.94902),("PF","PX",0.85567)]
        context.days=62
        context.R=20
        context.H=1
        context.sigma = 5
        context.target_vol=0.1
        context.direction=1
        context.fired=False
        context.typelist=["JM","J","RB","HC","SC","LU","L","PP","RU","NR","M","RM","Y","OI","P","PX","TA","PF"]
        context.count=0#用于计时
        context.range=0.2#取前20
        context.name=f"volbottomtargetvol_S{context.sigma}_T{context.target_vol}"
        context.length = len(context.hedge_choice)
        context.cummulative ={}
        for i in context.hedge_choice:
            context.cummulative[(i[0],i[1])]=0
        for item in context.typelist:
            self.subscribe(item)#注册品种
        #print(self.data)
    def before_trade(self, context,_data):
        if context.fired:
            context.count+=1       
        return super().before_trade(context)
    
    def handle_bar(self, m_data, context):
        for choice in context.hedge_choice:
            try:
                type1,type2,rho = choice[0],choice[1],choice[2]
                sigma1 ,sigma2 =m_data[type1]["sigma63"].iloc[-1],m_data[type2]["sigma63"].iloc[-1]
                if sigma1==0 or sigma2==0:
                    continue
                profit1,profit2 = m_data[type1]["profit"].iloc[-1],m_data[type2]["profit"].iloc[-1]
                proportion = profit1/sigma1-profit2*rho/sigma2
                

                if np.isnan(proportion):
                    continue
                if(profit1/sigma1>profit2*rho/sigma2):
                    direction = True
                else:
                    direction = False
                context.cummulative[(type1,type2)]= context.cummulative[(type1,type2)]*0.99+proportion
                
        
            except: continue

        if context.fired:
            if context.count<context.H:
                return
            else:
                self.sell_all_target(m_data)
                context.count=0
                context.fired = False
        if not context.fired:
            ##开始计算每个对冲组合的相关度

            cash_max = (self.position.cash//(2))/10000
            max_compound = 0
            max_key =()
            for key,value in context.cummulative.items():
                if(abs(value)>abs(max_compound)):
                    max_compound=value
                    max_key=key
            print(max_key)
            for choice in context.hedge_choice:
                if choice[0]!=max_key[0] or choice[1]!=max_key[1]:
                    continue
                type1,type2,rho = choice[0],choice[1],choice[2]
                sigma1 ,sigma2 =m_data[type1]["sigma63"].iloc[-1],m_data[type2]["sigma63"].iloc[-1]
                if sigma1==0 or sigma2==0:
                    return
                profit1,profit2 = m_data[type1]["profit"].iloc[-1],m_data[type2]["sigma63"].iloc[-1]
                
                if(max_compound>0):
                    direction = True
                else:
                    direction = False
                close1,close2 = m_data[type1]["close"].iloc[-1],m_data[type2]["close"].iloc[-1]
                multi1,multi2 = m_data[type1]["multiplier"].iloc[-1], m_data[type2]["multiplier"].iloc[-1]
                if int(cash_max/(close1*multi1))>0 and int(cash_max/(close2*multi2))>0:
                    print("here")
                    self.order_target_num(close1,int(cash_max/(close1*multi1)),multi1,type1,"short" if direction else "long")
                    self.order_target_num(close2,int(cash_max/(close2*multi2)),multi2,type2,"long" if direction else "short")
                    context.fired=True  
        
            

    def after_trade(self, context):
        pass
        
        
for h in [10,20]:
    # for j in []:
        
    engine = Section_Boost_BackTest(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
    engine.context.H=h
    engine.context.name=f"volhedge_H{engine.context.H}_T"
    engine.loop_process(start="20120101",end="20240501")
# engine = Section_Momentum_BackTest(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.name= "best_section"