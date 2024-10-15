# # #plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题  

file_name = "section/newsecbreakadvance/"
real_name = "newsecbreakmadvance"
tail =""
#新建dfs
indexes :dict={"W": [f"{i}" for i in [10,11,12,13]]}
columns ={"T": [f"{i:.1f}" for i in[1.5,2]]}


################################################################################# 

index_title = list(indexes.keys())[0]
column_title = list(columns.keys())[0]
plot_name =[]
plots = pd.read_excel("back/"+file_name+"Back_"+real_name+f"_{index_title}{indexes[index_title][0]}_{column_title}{columns[column_title][0]}{tail}.xlsx")
for r in indexes[index_title]:
    for s in columns[column_title]:
        name= real_name+f"_{index_title}{r}_{column_title}{s}{tail}"#需修改
        plot_name.append(name)
        plots[name] = pd.read_excel("back/"+file_name+"Back_"+name+".xlsx")[name]




plots.plot("date",plot_name,title="",grid=True)
plt.show()
# plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
# plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题        
# plots.plot("date",plot_name,title="横截面动量策略不同参数(R, H)收益曲线",grid=True,legend="lower center") 
# plt.legend(loc = (1.00,0))

# plt.show()       