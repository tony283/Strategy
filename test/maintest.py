import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
a= "data/CU_daily.xlsx"
df = pd.read_excel(a,index_col="date")

X=df.index
Y1=df["close"]
Y2=df["volume"]
fig, ax1 = plt.subplots()  
ax2 = ax1.twinx()  
df =df[["close","volume","profit"]]
df["1"]=None
df["2"]=None
df["3"]=None
df["4"]=None
last_volume=0
last_profit=0
"四阶段:价升量升，价升量跌,价跌量升，价跌量跌，"
for index, row in df.iterrows():
    volume,profit = row["volume"],row["profit"]
    
    stage=2
    if profit<0 and last_profit>0.025:
        stage=1
    df.loc[index,f"{stage}"]=row["close"]
    last_profit,last_volume=profit,volume
    
print(df)


#绘制折线图  
line1 = ax1.plot(X, Y1,label='y1轴', color='royalblue') 
line3 = ax1.scatter(X,df["1"],color="red") 
line2 = ax2.bar(X, Y2, label='y2轴', color='tomato')  
ax2.set_ylim((0,5e6))
# 设置x轴和y轴的标签，指明坐标含义  
ax1.set_xlabel('x轴', fontdict={'size': 16})  
ax1.set_ylabel('y1轴',fontdict={'size': 16})  
ax2.set_ylabel('y2轴',fontdict={'size': 16})  
#添加图表题  
plt.title('双y轴折线图')  
#添加图例  
plt.legend()  
# 设置中文显示  
plt.rcParams['font.sans-serif']=['SimHei']  
#展示图片  
plt.show()