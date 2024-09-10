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
        if context.fired:
            if context.count<context.H:
                return
            else:
                self.sell_all_target(m_data)
                context.count=0
                context.fired=False
        if not context.fired:
            ##开始计算每个品种的绝对收益率
            temp_dict =[]#用于储存收益率信息
            for future_type in context.typelist:
                try:
                    profit_max = m_data[future_type].iloc[-context.R:].copy()
                    profit_max["abs_profit"]= profit_max["profit"].apply(lambda x : abs(x))
                    
                    close_mean = profit_max["close"].mean()
                    profit_max = profit_max["abs_profit"].max()
                    if profit_max==0 or profit_max>context.profit:
                        continue
                    close =m_data[future_type]["close"].iloc[-1]
                    direction = "long" if close<=close_mean else "short"
                    temp_dict.append([future_type,direction])
                except:
                    continue
            ranking = pd.DataFrame(temp_dict,columns=["future_type","direction"])
            print(self.current)
            print(ranking)
            cash_max = (self.position.cash//(len(ranking)))//10000
            for index, row in ranking.iterrows():
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                
                buy_amount = int(cash_max/(close*multi))
                if buy_amount<=0:
                    continue
                self.order_target_num(close,buy_amount,multi,future_type,row["direction"])

                context.fired=True

    def after_trade(self, context):
        pass
        
        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for n in [i for i in range(16,19)]:
        for h in [0.010]:#[0.008,0.010,0.012,0.014,0.016,0.018,0.02]:
            engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.R=n
            engine.context.profit=h
            engine.context.H=2
            engine.context.name = f"lowvol_Profit{h:.3f}_R{n}"
            p.apply_async(engine.loop_process,args=("20120101","20240501","back/vol/lowvol/"))
            # engine.loop_process("20120101","20240501",saving_dir="back/vol/lowvol/")
    print("-----start-----")
    p.close()
    p.join()
    print("------end------")
# engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.R=20

# engine.context.H=20
# engine.context.name = f"test"
# engine.loop_process(start="20150101",end="20231231",saving_dir="back/")