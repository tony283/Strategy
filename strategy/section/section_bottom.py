import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#sys.path.append("c:\\Users\\ROG\\Desktop\\Strategy\\utils")
import os
import multiprocessing
from utils.BackTestEngine import *
#order_target_num 用来下空单或多单
#sell_target_num 用来平仓
class Section_Bottom_BackTest(BackTest):
    def init(self, context):
        context.days=62
        context.R=20
        context.H=5
        context.sigma = 5
        context.target_vol=0.1
        context.direction=True
        context.fired=False
        context.typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
        context.count=0#用于计时
        context.range=0.2#取前20
        context.name=f"voltoptargetvol_S{context.sigma}_T{context.target_vol}_day{context.days}"
        for item in context.typelist:
            self.subscribe(item)#注册品种
        #print(self.data)
    def before_trade(self, context,m_data):
        if context.fired:
            context.count+=1
        if(len(self.position.asset)<65):
            return
        if((self.position.asset[-2]-self.position.asset[-context.days-2])/self.position.asset[-context.days-2]<-1.5*np.sqrt((context.days)/252)*context.target_vol):
            context.direction= not context.direction#切换方向
            
        
        
        
        return super().before_trade(context)
    
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
                    profit = (m_data[future_type]["close"].iloc[-2]-m_data[future_type]["close"].iloc[-2-context.R])/m_data[future_type]["close"].iloc[-2-context.R]
                    sigma = m_data[future_type]["sigma"+str(context.sigma)].iloc[-1]
                    temp_dict.append([future_type,profit,sigma])
                    
                except:
                    continue
            ranking = pd.DataFrame(temp_dict,columns=["future_type","profit","sigma"])
            range=int(self.context.range*len(ranking))
            ranking = ranking.sort_values(by="profit",ascending=True)#排名
            ranking = ranking.iloc[:range]
            ranking = ranking.sort_values(by="sigma",ascending=True)
            range = 1
            cash_max = (self.position.cash//(range))/10000
            # for index, row in ranking.iloc[-range:].iterrows():#收益率最高的
            #     future_type=row["future_type"]
            #     close = m_data[future_type]["close"].iloc[-1]
            #     multi = m_data[future_type]["multiplier"].iloc[-1]
            #     self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"long")
            for index, row in ranking.iloc[:range].iterrows():#收益率最高且波动率最低的
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                self.order_target_num(close,int(cash_max/(close*multi)),multi,future_type,"long" if context.direction else "short")#自动切换
            context.fired=True

    def after_trade(self, context):
        pass
        
        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for t in [0.05,0.1,0.15,0.2,0.25,0.3]:
        for s in [5,20,40,63,126,252]:
            engine = Section_Bottom_BackTest(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.sigma=s
            engine.context.target_vol = t
            engine.context.name=f"voltoptargetvol_S{s}_T{t}_day62"
            p.apply_async(engine.loop_process,args=("20120101","20240501"))
    print("-----start-----")
    p.close()
    p.join()
    print("------end------")
        #engine.loop_process(start="20120101",end="20240501")
        
# engine = Section_Momentum_BackTest(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.name= "best_section"




#获取收益因子
# engine = Section_Boost_BackTest(cash=100000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.sigma=20
# engine.context.target_vol = 0.25
# engine.context.name=f"volbottomtargetvol_S{s}_T{t}_day62"
# engine.loop_process("20120101","20240501")