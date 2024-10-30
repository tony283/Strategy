import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
# 创建示例数据
data = np.zeros((51, 51))  # 10x10 的随机数据
model = joblib.load(f'data/RF_Data/random_forest_model_v1_0_2_{4}.pkl')
max=4


step=max/50
for i in range(51):
    for j in range(51):
        data[i,j]=model.predict(pd.DataFrame([[step*(i-25),0,0,step*(j-25),0]],columns=[f'break{i}' for i in [3,14,20,63,126]]))[0]

# 创建热力图
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()  # 添加颜色条
plt.title('Heatmap using imshow')
locs, labels = plt.xticks()
print(locs)
plt.xticks(ticks=np.arange(8),labels=np.linspace(-max/2,max/2,8))
plt.yticks(ticks=np.arange(8),labels=np.linspace(-max/2,max/2,8))
plt.show()