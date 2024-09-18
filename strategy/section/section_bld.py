import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import time
sys.path.append("strategy/")
import os
import multiprocessing
from utils.BackTestEngine import *
# from BackTestEngineC import *


#order_target_num 用来下空单或多单
#sell_target_num 用来平仓
class Section_Momentum_BackTest(BackTest):
    def init(self, context):
        context.R=20
        context.H=20
        context.name=f"section_R{context.R}_H{context.H}"
        context.fired=False
        context.typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
        context.count=0#用于计时
        for item in context.typelist:
            self.subscribe(item)#注册品种
        #print(self.data)
    def before_trade(self, context,m_data):
        if context.fired:
            context.count+=1
    def handle_bar(self, m_data, context):
        timer1 = time.time()
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
                    self.sell_target_num(int(m_data[future_type]["close"].iloc[-1]),int(amount[0]//multi),int(multi),future_type,direction)
                    context.count=0
                    context.fired=False

        if not context.fired:
            ##开始计算每个品种的收益率
            temp_dict =[]#用于储存收益率信息
            for future_type in context.typelist:
                try:
                    profit = (m_data[future_type]["close"].iloc[-1]-m_data[future_type]["close"].iloc[-1-context.R])/m_data[future_type]["close"].iloc[-1-context.R]
                    sigma = m_data[future_type]["profit"].iloc[-context.N:].std()
                    liquidity = m_data[future_type].iloc[-context.L:].copy()
                    liquidity["liquidity"] = np.log(liquidity["profit"].apply(lambda x:abs(x))/(liquidity["close"]*liquidity["volume"]/1e11)+1)
                    liquidity = liquidity["liquidity"].mean()
                    temp_dict.append([future_type,profit,sigma,liquidity])
                except:
                    continue
            ranking = pd.DataFrame(temp_dict,columns=["future_type","profit","sigma","liquidity"])
            ranking = ranking[ranking["sigma"]!=0]
            ranking["break"] = ranking["profit"].apply(lambda x:abs(x))/(ranking['sigma']*np.sqrt(context.R))
            ranking["BLD"]=ranking['break']*ranking['liquidity']
            ranking = ranking.replace([np.inf, -np.inf], np.nan).dropna()
            cash_max = (self.position.cash)//10000
            sum_break = ranking["BLD"].sum()
            for index, row in ranking.iterrows():#多空
                future_type=row["future_type"]
                proportion = row["BLD"]/sum_break
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                
                buy_amount = int(cash_max*proportion/(close*multi))
                if row["profit"]==0:
                    continue
                if buy_amount<=0:
                    continue
                self.order_target_num(close,int(buy_amount),int(multi),future_type,"long" if row["profit"]>0 else "short")
                context.fired=True


    def after_trade(self, context):
        pass
        
        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for n in [1,2,3,4,5]:
        for h in range(1,18):
            engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.R=19
            engine.context.N=20
            engine.context.H=n
            engine.context.L=h
            engine.context.name = f"newsecbld_Liquidity{h}_H{n}"
            # p.apply_async(engine.loop_process,args=("20120101","20240501","back/section/newsecbreakall/"))
            engine.loop_process(start="20150101",end="20231231",saving_dir="back/section/newsecbld/")
    print("-----start-----")
    p.close()
    p.join()
    print("------end------")
# engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.R=20

# engine.context.H=20
# engine.context.name = f"test"
# engine.loop_process(start="20150101",end="20231231",saving_dir="back/")