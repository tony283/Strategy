import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import warnings
# plot_name = ["section_R"+str(5*i)+"_H"+str(20) for i in range(1,9)]
# plots={}
# for i in plot_name:
#     plots[i]=pd.read_excel("back/Back_"+i+".xlsx")
    
# total:pd.DataFrame = plots[plot_name[0]]
# for i in range(len(plot_name)-1):
#     total[plot_name[i+1]]=plots[plot_name[i+1]][plot_name[i+1]]


# total.plot("date",plot_name,title="横截面动量策略不同参数(R, H)收益曲线",grid=True)
# plt.show()
factors=['sigma5', 'sigma20', 'sigma40', 'sigma63', 'sigma126', 'sigma252', 'break1', 'break3', 'break14', 'break20', 'break63', 'break126', 'break252','d_vol', 'd_oi', 'mmt_open', 'high_close', 'low_close', 'corr_price_vol', 'corr_price_oi', 'corr_ret_vol', 'corr_ret_oi', 'corr_ret_dvol', 'corr_ret_doi', 'turnover', 'sigma_turnover', 'ave_turnover', 'norm_turn_std', 'vol_skew5', 'vol_skew14', 'vol_skew20', 'vol_skew63', 'vol_skew126', 'vol_skew252', 'price_skew5', 'price_skew14', 'price_skew20', 'price_skew63', 'price_skew126', 'price_skew252', 'sigma_skew5', 'sigma_skew14', 'sigma_skew20', 'sigma_skew63', 'sigma_skew126', 'sigma_skew252', 'low_close_high', 'd_low_close_high', 'mean6', 'mean12', 'dif', 'dea', 'macd', 'sma_low_close_high9', 'sma_low_close_high6', 'std_vol6', 'ddif_vol', 'norm_ATR', 'sq5_low_close_open_high', 'vol_kurt5', 'vol_kurt14', 'vol_kurt20', 'vol_kurt63', 'vol_kurt126', 'vol_kurt252', 'price_kurt5', 'price_kurt14', 'price_kurt20', 'price_kurt63', 'price_kurt126', 'price_kurt252', 'sigma_kurt5', 'sigma_kurt14', 'sigma_kurt20', 'sigma_kurt63', 'sigma_kurt126', 'sigma_kurt252', 'winrate5', 'winrate20', 'winrate63', 'winrate126', 'draw5', 'draw20', 'draw63', 'draw126', 'position5', 'position20', 'position63', 'position126', 'd_position5', 'd_position20', 'd_position63', 'daily_position5', 'daily_position20', 'd_daily_position', 'relative_amihud5', 'highlow_avg5', 'highlow_std5', 'upshadow_avg5', 'upshadow_std5', 'downshadow_avg5', 'relative_amihud20', 'highlow_avg20', 'highlow_std20', 'upshadow_avg20', 'upshadow_std20', 'downshadow_avg20', 'relative_amihud63', 'highlow_avg63', 'highlow_std63', 'upshadow_avg63', 'upshadow_std63', 'downshadow_avg63', 'relative_amihud126', 'highlow_avg126', 'highlow_std126', 'upshadow_avg126', 'upshadow_std126', 'downshadow_avg126','high_m_low','MAX(close-SMA(close,5))','d_position','skew_position5','skew_position20','skew_position63','skew_position126','sigma_skew20_m_position63','sigma_skew20_m_d_position5']

file_name = "section/newsecfactor/"
real_name = "newsecfactor"
tail =""
#新建dfs
indexes :dict={"": [f"{i}" for i in factors]}
columns ={"H": [f"{i}" for i in range(1,4)]}


#以上为需要填的参数
#-------------------------------------------------------------------------------------------------------------------------------------------------------#

index_title = list(indexes.keys())[0]
column_title = list(columns.keys())[0]
df_profit = pd.DataFrame(index=[f"{index_title}{i}" for i in indexes[index_title]],columns=[f"{column_title}{i}" for i in columns[column_title]])
df_sigma = pd.DataFrame(index=[f"{index_title}{i}" for i in indexes[index_title]],columns=[f"{column_title}{i}" for i in columns[column_title]])
df_pro_div_sigma=  pd.DataFrame(index=[f"{index_title}{i}" for i in indexes[index_title]],columns=[f"{column_title}{i}" for i in columns[column_title]])
df_maxdrawdownrate = pd.DataFrame(index=[f"{index_title}{i}" for i in indexes[index_title]],columns=[f"{column_title}{i}" for i in columns[column_title]])
plots = pd.read_excel("back/"+file_name+"Back_"+real_name+f"_{index_title}{indexes[index_title][0]}_{column_title}{columns[column_title][0]}{tail}.xlsx")
T=len(plots)/252


plot_name=[]
for r in indexes[index_title]:
    for s in columns[column_title]:
        name= real_name+f"_{index_title}{r}_{column_title}{s}{tail}"#需修改
        plots[name] = pd.read_excel("back/"+file_name+"Back_"+name+".xlsx")[name]
        plot_name.append(name)
        profit = pow((plots[name].iloc[-1]/plots[name].iloc[0]),1/T)-1
        plots["profit"]=np.log(plots[name].shift(-1)/plots[name])
        sigma = plots["profit"].std()*np.sqrt(252)
        pro_div_sigma = profit/sigma
        plots["drawdown"]=(plots[name].cummax()-plots[name])/plots[name].cummax()
        maxdrawdownrate= plots["drawdown"].max()
        calmar =profit/maxdrawdownrate
        df_profit.loc[f"{index_title}{r}",f"{column_title}{s}"]=profit
        df_sigma.loc[f"{index_title}{r}",f"{column_title}{s}"]=sigma
        df_maxdrawdownrate.loc[f"{index_title}{r}",f"{column_title}{s}"]=calmar
        df_pro_div_sigma.loc[f"{index_title}{r}",f"{column_title}{s}"]=pro_div_sigma    
        
if not os.path.exists("Report/"+file_name):
    os.makedirs("Report/"+file_name)
    print(f"Folder created: Report/{file_name}")
print('profit')
print(df_profit)
print('calmar')
print(df_maxdrawdownrate)
print("sharpe")
print(df_pro_div_sigma)
df_profit.to_excel("Report/"+file_name+real_name+"_profit.xlsx")
df_sigma.to_excel("Report/"+file_name+real_name+"_sigma.xlsx")
df_maxdrawdownrate.to_excel("Report/"+file_name+real_name+"_calmar.xlsx")
df_pro_div_sigma.to_excel("Report/"+file_name+real_name+"_pro_div_sigma.xlsx")


# # #plt
# # plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
# # plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题        
# # plots.plot("date",plot_name,title="横截面动量策略不同参数(R, H)收益曲线",grid=True,legend="lower center") 
# # plt.legend(loc = (1.00,0))

# # plt.show()       

