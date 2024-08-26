import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
# plot_name = ["section_R"+str(5*i)+"_H"+str(20) for i in range(1,9)]
# plots={}
# for i in plot_name:
#     plots[i]=pd.read_excel("back/Back_"+i+".xlsx")
    
# total:pd.DataFrame = plots[plot_name[0]]
# for i in range(len(plot_name)-1):
#     total[plot_name[i+1]]=plots[plot_name[i+1]][plot_name[i+1]]


# total.plot("date",plot_name,title="横截面动量策略不同参数(R, H)收益曲线",grid=True)
# plt.show()

file_name = "convex/convex+volume/"
real_name = "convex+volume"
T=2267/252
#新建df
df_profit = pd.DataFrame(index=[f"N{i}" for i in [10,20,60,120]],columns=[f"M{i}" for i in [0,1,2,5]])
df_sigma = pd.DataFrame(index=[f"N{i}" for i in [10,20,60,120]],columns=[f"M{i}" for i in [0,1,2,5]])
df_pro_div_sigma=  pd.DataFrame(index=[f"N{i}" for i in [10,20,60,120]],columns=[f"M{i}" for i in [0,1,2,5]])
df_maxdrawdownrate = pd.DataFrame(index=[f"N{i}" for i in [10,20,60,120]],columns=[f"M{i}" for i in [0,1,2,5]])
plots = pd.read_excel("back/"+file_name+"Back_"+real_name+"_N20_M0.xlsx")
T=len(plots)/252


plot_name=[]
for n in [10,20,60,120]:
    for m in [0,1,2,5]:
        name= real_name+f"_N{n}_M{m}"#需修改
        plots[name] = pd.read_excel("back/"+file_name+"Back_"+name+".xlsx")[name]
        plot_name.append(name)
        profit = pow((plots[name].iloc[-1]/plots[name].iloc[0]),1/T)-1
        plots["profit"]=np.log(plots[name].shift(-1)/plots[name])
        sigma = plots["profit"].std()*np.sqrt(252)
        pro_div_sigma = profit/sigma
        plots["drawdown"]=(plots[name].cummax()-plots[name])/plots[name].cummax()
        maxdrawdownrate= plots["drawdown"].max()
        df_profit.loc[f"N{n}",f"M{m}"]=profit
        df_sigma.loc[f"N{n}",f"M{m}"]=sigma
        df_maxdrawdownrate.loc[f"N{n}",f"M{m}"]=maxdrawdownrate
        df_pro_div_sigma.loc[f"N{n}",f"M{m}"]=pro_div_sigma    
        
if not os.path.exists("Report/"+file_name):
    os.makedirs("Report/"+file_name)
    print("Folder created")
df_profit.to_excel("Report/"+file_name+real_name+"_profit.xlsx")
df_sigma.to_excel("Report/"+file_name+real_name+"_sigma.xlsx")
df_maxdrawdownrate.to_excel("Report/"+file_name+real_name+"_drawdown.xlsx")
df_pro_div_sigma.to_excel("Report/"+file_name+real_name+"_pro_div_sigma.xlsx")


# #plt
# plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
# plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题        
# plots.plot("date",plot_name,title="横截面动量策略不同参数(R, H)收益曲线",grid=True,legend="lower center") 
# plt.legend(loc = (1.00,0))

# plt.show()       