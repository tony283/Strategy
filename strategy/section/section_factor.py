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
        # joblib.dump(self.data, f'data/alldata.pkl')
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
                    factor = m_data[future_type][context.factor].iloc[-1]
                    temp_dict.append([future_type,factor])
                except:
                    continue
            ranking = pd.DataFrame(temp_dict,columns=["future_type","factor"])
            ranking=ranking.dropna()
            range=int(self.context.range*len(ranking))
            ranking = ranking.sort_values(by="factor",ascending=True)#排名
            
            cash_max = (self.position.cash//(2*range))/10000
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
        
factors=['sigma5', 'sigma20', 'sigma40', 'sigma63', 'sigma126', 'sigma252', 'break1', 'break3', 'break14', 'break20', 'break63', 'break126', 'break252','d_vol', 'd_oi', 'mmt_open', 'high_close', 'low_close', 'corr_price_vol', 'corr_price_oi', 'corr_ret_vol', 'corr_ret_oi', 'corr_ret_dvol', 'corr_ret_doi', 'turnover', 'sigma_turnover', 'ave_turnover', 'norm_turn_std', 'vol_skew5', 'vol_skew14', 'vol_skew20', 'vol_skew63', 'vol_skew126', 'vol_skew252', 'price_skew5', 'price_skew14', 'price_skew20', 'price_skew63', 'price_skew126', 'price_skew252', 'sigma_skew5', 'sigma_skew14', 'sigma_skew20', 'sigma_skew63', 'sigma_skew126', 'sigma_skew252', 'low_close_high', 'd_low_close_high', 'mean6', 'mean12', 'dif', 'dea', 'macd', 'sma_low_close_high9', 'sma_low_close_high6', 'std_vol6', 'ddif_vol', 'norm_ATR', 'sq5_low_close_open_high', 'vol_kurt5', 'vol_kurt14', 'vol_kurt20', 'vol_kurt63', 'vol_kurt126', 'vol_kurt252', 'price_kurt5', 'price_kurt14', 'price_kurt20', 'price_kurt63', 'price_kurt126', 'price_kurt252', 'sigma_kurt5', 'sigma_kurt14', 'sigma_kurt20', 'sigma_kurt63', 'sigma_kurt126', 'sigma_kurt252', 'winrate5', 'winrate20', 'winrate63', 'winrate126', 'draw5', 'draw20', 'draw63', 'draw126', 'position5', 'position20', 'position63', 'position126', 'd_position5', 'd_position20', 'd_position63', 'daily_position5', 'daily_position20', 'd_daily_position', 'relative_amihud5', 'highlow_avg5', 'highlow_std5', 'upshadow_avg5', 'upshadow_std5', 'downshadow_avg5', 'relative_amihud20', 'highlow_avg20', 'highlow_std20', 'upshadow_avg20', 'upshadow_std20', 'downshadow_avg20', 'relative_amihud63', 'highlow_avg63', 'highlow_std63', 'upshadow_avg63', 'upshadow_std63', 'downshadow_avg63', 'relative_amihud126', 'highlow_avg126', 'highlow_std126', 'upshadow_avg126', 'upshadow_std126', 'downshadow_avg126','high_m_low','MAX(close-SMA(close,5))','d_position','skew_position5','skew_position20','skew_position63','skew_position126','sigma_skew20_m_position63','sigma_skew20_m_d_position5','ADD[d_position5 , PROD[vol_skew126 , skew_position63]]','DIF5(skew_position63)','RANK9(skew_position63)','ADD[skew_position20 , position63]','PROD[RANK26(vol_kurt126) , low_close]','MINUS[skew_position63 , relative_amihud5]']        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for n in list(['PROD[RANK26(vol_kurt126) , low_close]','MINUS[skew_position63 , relative_amihud5]']):
        for h in range(1,4):
            engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.H=h
            engine.context.range = 0.1
            engine.context.factor=n
            engine.context.name = f"newsecfactor_{n}_H{h}"
            p.apply_async(engine.loop_process,args=("20180101","20241030","back/section/newsecfactor/"))
            # engine.loop_process(start="20200101",end="20241030",saving_dir="back/section/newsecfactor/")
    p.close()
    p.join()
# engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.R=20

# engine.context.H=20
# engine.context.name = f"test"
# engine.loop_process(start="20150101",end="20231231",saving_dir="back/")