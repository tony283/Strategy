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



file_name = "section/newsecbreakadvance/"
real_name = "newsecbreakmadvance"
tail =""
#新建dfs
indexes :dict={"W": [f"{i}" for i in [2,3,4,5,6,7,8,9,10,11,12,13]]}
columns ={"T": [f"{i:.1f}" for i in[1,1.5,2,2.5,3]]}


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
    print("Folder created")
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