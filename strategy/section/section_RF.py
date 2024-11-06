'''
:@Author: LRF
:@Date: 11/5/2024, 3:33:33 PM
:@LastEditors: LRF
:@LastEditTime: 11/5/2024, 3:33:33 PM
:Description: 
'''
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("strategy/")
import os
import multiprocessing
from utils.BackTestEngine import *
import joblib
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
                    
 
                    temp_dict.append([future_type,m_data[future_type]["sigma5"].iloc[-1]])
                except:
                    continue
            ranking = pd.DataFrame(temp_dict,columns=["future_type","sigma"])
            ranking = ranking.dropna()
            range=int(self.context.range*len(ranking))
            ranking = ranking.sort_values(by="sigma",ascending=False)#排名
            cash_max = (self.position.cash//(range))/10000
            
            for index, row in ranking.iloc[:range].iterrows():#收益率最低的
                future_type=row["future_type"]
                try:
                    s = m_data[future_type][["break1","break3",'break14','break20','break63','break126','d_vol','d_oi','high_close','low_close','corr_price_vol','corr_price_oi','corr_ret_oi','corr_ret_dvol','corr_ret_doi','norm_turn_std','vol_skew5','vol_skew14','vol_skew252','price_skew5','price_skew14','price_skew20','price_skew126']].iloc[-1].to_numpy()
                    if (True in np.isnan(s)):
                        continue

                    y_pred=context.loaded_model.predict(pd.DataFrame([s],columns=["break1","break3",'break14','break20','break63','break126','d_vol','d_oi','high_close','low_close','corr_price_vol','corr_price_oi','corr_ret_oi','corr_ret_dvol','corr_ret_doi','norm_turn_std','vol_skew5','vol_skew14','vol_skew252','price_skew5','price_skew14','price_skew20','price_skew126']))
                except:
                    continue
                if not (y_pred[0]==0 or y_pred[0]==1):
                    continue
                close = m_data[future_type]["close"].iloc[-1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                # usage=row["usage"]
                buy_amount = int(cash_max/(close*multi))
                if(buy_amount<=0):
                    continue
                
                self.order_target_num(close,buy_amount,multi,future_type,"short" if y_pred[0]==0 else 'long')
                context.fired=True

    def after_trade(self, context):
        pass
        
        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for n in [0.05,0.1,0.15,0.2,0.25]:
        for h in range(1,6):
            engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.H=h
            engine.context.range = n
            engine.context.name = f"newsecXGBv142_Rg{n:.2f}_H{h}"
            engine.context.loaded_model = joblib.load(f'data/RF_Data/XGBoost_v1_4_2_{h}.pkl')
            p.apply_async(engine.loop_process,args=("20180103","20241103","back/section/newsecXGB/"))
            # engine.loop_process(start="20180101",end="20240501",saving_dir="back/section/newsecXGB/")
    # print("-----start-----")
    p.close()
    p.join()
    # print("------end------")
# engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.R=20

# engine.context.H=20
# engine.context.name = f"test"
# engine.loop_process(start="20150101",end="20231231",saving_dir="back/")
