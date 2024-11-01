import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("strategy/")
import os
import multiprocessing
from utils.BackTestEngine import *
import joblib
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
        context.update_freq_count=9999
        context.range=0.2#取前20%
        for item in context.typelist:
            self.csv_subscribe(item)#注册品种

        self.vol = pd.read_excel("data/future_std.xlsx",index_col=0)
        
        #print(self.data)
    def before_trade(self, context,m_data):
        context.update_freq_count+=1
        if context.fired:
            context.count+=1
    
    def handle_bar(self, m_data, context):
        if context.update_freq_count>context.update_freq:
            context.update_freq_count=0
            breaklist=["break3",'break14','break20','break63','break126','expect1','expect2','expect3','expect4','expect5']
            df=pd.DataFrame(columns=breaklist)
            for i in m_data.values():
                if len(i)<5:
                    continue
                if len(i)>504:
                    if len(df)==0:
                        df =i[breaklist].iloc[-504:-5]
                        continue
                    df=pd.concat([df,i[breaklist].iloc[-504:-5]])
                else:
                    if len(df)==0:
                        df =i[breaklist].iloc[:-5]
                        continue
                    df=pd.concat([df,i[breaklist].iloc[:-5]])
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df=df.dropna()
            self.model = self.UpdateModel(m_data,df,context.H)
            
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
                    s = m_data[future_type]["sigma20"].iloc[-1]
                    breaklist=m_data[future_type][["break3",'break14','break20','break63','break126']].iloc[-1].to_numpy()
                    breaklist=pd.DataFrame([breaklist],columns=[f'break{i}' for i in [3, 14, 20, 63, 126]])

                    y_pred=self.model.predict(breaklist)

                    if(s==0 or s!=s or m_data[future_type]["close"].iloc[-127]==0):
                        continue
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
    def UpdateModel(self,m_data,df,H):
        X_train = df[[f'break{i}' for i in [3,14,20,63,126]]]  # 替换为你的特征列
        y_train = df[f'expect{H}'].apply(lambda x: 1 if x>0 else 0)

        # 拆分数据集为训练集和测试集
        
        param_grid = {
        'objective': ['binary:logistic'],  # For binary classification
        'n_estimators': [150],
        'max_depth': [10],
        'learning_rate': [0.1],
        'gamma': [0],
        }
        # 创建随机森林分类器
        model = xgb.XGBClassifier(eval_metric='mlogloss', device='cuda')
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                            scoring='f1', cv=5, verbose=0)
        grid_search.fit(X_train, y_train)
        # 训练模型
        return grid_search.best_estimator_
        
        
        
if(__name__=="__main__"):
    p=multiprocessing.Pool(40)
    for n in [20,40,63,126]:
        for h in range(1,6):
            engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
            engine.context.H=h
            engine.context.range = 0.1
            engine.context.update_freq=n
            engine.context.name = f"newsecXGBv101_Freq{n}_H{h}"
            # p.apply_async(engine.loop_process,args=("20180101","20241030","back/section/newsecXGB/"))
            engine.loop_process(start="20180201",end="20241030",saving_dir="back/section/newsecXGB/")
    # print("-----start-----")
    p.close()
    p.join()
    # print("------end------")
# engine = Section_Momentum_BackTest(cash=1000000000,margin_rate=1,margin_limit=0,debug=False)
# engine.context.R=20

# engine.context.H=20
# engine.context.name = f"test"
# engine.loop_process(start="20150101",end="20231231",saving_dir="back/")
