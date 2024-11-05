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
        context.importance=[]
        context.feature=[]
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
            temp_list=[]#用于计算PCA
            for future_type in context.typelist:
                try:
                    s = m_data[future_type]["sigma20"].iloc[-1]
                    if(s==0 or m_data[future_type]["close"].iloc[-126-1]==0):
                        continue
                    breaklist=[(m_data[future_type]["close"].iloc[-1]-m_data[future_type]["close"].iloc[-R-1])/(s*m_data[future_type]["close"].iloc[-R-1]*np.sqrt(R)) for R in [3,14,20,63,126]]
                    temp_list.append(breaklist)
                except:
                    continue
            X=np.array(temp_list).T
            corr=X@X.T
            
            eigenvalue, featurevector = np.linalg.eig(corr)
            context.importance.append(eigenvalue/eigenvalue.sum())
            main_feature=featurevector[np.argmax(eigenvalue)]
            main_feature=np.sign(main_feature.sum())*main_feature
            context.feature.append(main_feature)
            for future_type in context.typelist:
                try:
                    s = m_data[future_type]["sigma20"].iloc[-1]
                    if(s==0 or m_data[future_type]["close"].iloc[-126-1]==0):
                        continue
                    breaklist=[(m_data[future_type]["close"].iloc[-1]-m_data[future_type]["close"].iloc[-R-1])/(s*m_data[future_type]["close"].iloc[-R-1]*np.sqrt(R)) for R in [3,14,20,63,126]]
                    PCA=main_feature@np.array(breaklist)
                    temp_dict.append([future_type,PCA,s])
                except:
                    continue
            ranking = pd.DataFrame(temp_dict,columns=["future_type","VWAP","sigma"])
            ranking = ranking[ranking["VWAP"]!=0]
            ranking = ranking.dropna()
            ranking['rs']=1/ranking["sigma"]
            ranking["usage"] = ranking["sigma"].apply(lambda x:pow(min(0.016/x,1),2))
            # ranking=ranking[ranking["break"]!=0]
            ranking=ranking[ranking["usage"]!=0]
            range=int(self.context.range*len(ranking))
            ranking = ranking.sort_values(by="VWAP",ascending=True)#排名
            cash_max = (self.position.cash//(2*range))/10000
            for index, row in ranking.iloc[-range:].iterrows():#收益率最高的
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                # usage=row["usage"]
                buy_amount = int(cash_max/(close*multi))
                if buy_amount<=0:
                    continue
                self.order_target_num(close,buy_amount,multi,future_type,"long")
            for index, row in ranking.iloc[:range].iterrows():#收益率最低的
                future_type=row["future_type"]
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                # usage=row["usage"]
                buy_amount = int(cash_max/(close*multi))
                if(buy_amount<=0):
                    continue
                self.order_target_num(close,buy_amount,multi,future_type,"short")
            context.fired=True

    def after_back(self, context):
        df_importance=pd.DataFrame(context.importance)
        df_feature=pd.DataFrame(context.feature)
        df_importance.to_csv(f"back/section/newsecPCA/{context.name}_IMP.csv")
        df_feature.to_csv(f"back/section/newsecPCA/{context.name}_FEATURE.csv")
        
        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for n in [0.05,0.1,0.15,0.2]:
        for h in range(1,6):
            engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.H=h
            engine.context.range = n
            engine.context.name = f"newsecPCA_Rg{n:.3f}_H{h}"
            p.apply_async(engine.loop_process,args=("20180101","20241030","back/section/newsecPCA/"))
            
            # engine.loop_process(start="20180101",end="20241030",saving_dir="back/section/newsecPCA/")
    print("-----start-----")
    p.close()
    p.join()
    print("------end------")
# engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.R=20

# engine.context.H=20
# engine.context.name = f"test"
# engine.loop_process(start="20150101",end="20231231",saving_dir="back/")