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
        context.range=0.2#取前20%
        for item in context.typelist:
            self.csv_subscribe(item)#注册品种
        self.vol = pd.read_excel("data/future_std.xlsx",index_col=0)
        #print(self.data)
    def before_trade(self, context,m_data):
        if context.fired:
            context.count+=1
    
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
                    temp_data=m_data[future_type]["profit"].iloc[-context.R:]
                    try:
                        high=temp_data[temp_data>0].sum()
                        low=-temp_data[temp_data<0].sum()
                        if high!=high:
                            high=0.0005
                        if low!=low or low==0:
                            low=0.0005
                        RS=high/low
                    except:
                        continue
                    RSI=100-100/(1+RS)
                    # sigma = m_data[future_type]["profit"].iloc[-context.N:].std()
                    # profitM = (m_data[future_type]["close"].iloc[-1]-m_data[future_type]["close"].iloc[-1-context.M])/m_data[future_type]["close"].iloc[-1-context.M]
                    temp_dict.append([future_type,RSI])
                except:
                    continue
            
            ranking = pd.DataFrame(temp_dict,columns=["future_type","RSI"])
            # ranking = ranking[ranking["sigma"]!=0]
            # ranking["break"] = ranking["M"].apply(lambda x:abs(x))/(ranking['sigma']*np.sqrt(context.M))
            # ranking["usage"] = ranking["sigma"].apply(lambda x:min(context.S/x,1))
            # ranking=ranking[ranking["break"]!=0]
            ranking=ranking[ranking["RSI"]!=50]
            ranking=ranking[ranking["RSI"]!=0]
            range=int(self.context.range*len(ranking))
            ranking = ranking.sort_values(by="RSI",ascending=True)#排名
            cash_max = (self.position.cash//(2*range))/10000
            # highest = ranking.iloc[-range:]["break"].sum()
            # lowest = ranking.iloc[:range]["break"].sum()
            for index, row in ranking.iloc[-range:].iterrows():#收益率最高的
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                buy_amount = int(cash_max/(close*multi))
                if buy_amount<=0:
                    continue
                self.order_target_num(close,buy_amount,multi,future_type,"long")
            for index, row in ranking.iloc[:range].iterrows():#收益率最低的
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                buy_amount = int(cash_max/(close*multi))
                if(buy_amount<=0):
                    continue
                self.order_target_num(close,buy_amount,multi,future_type,"short")
            context.fired=True

    def after_trade(self, context):
        pass
        
        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for n in range(7,15):
        for h in range(1,6):
            engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.R=n
            engine.context.H=h
            engine.context.range = 0.2
            engine.context.name = f"newsecRSI_R{n}_H{h}"
            p.apply_async(engine.loop_process,args=("20180101","20241030","back/section/newsecRSI/"))
            # engine.loop_process(start="20120101",end="20231231",saving_dir="back/section/newsecRSI/")
    # print("-----start-----")
    p.close()
    p.join()
    # print("------end------")
# engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.R=20

# engine.context.H=20
# engine.context.name = f"test"
# engine.loop_process(start="20150101",end="20231231",saving_dir="back/")