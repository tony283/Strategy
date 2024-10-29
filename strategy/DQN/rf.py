import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
# 假设你有一个CSV文件包含你的数据
# 读取数据
# data = pd.read_excel('data/RF_Data/rf.xlsx')

# # 假设你的数据集有以下列：'feature1', 'feature2', ..., 'featureN' 和 'target'
# # 其中 'target' 是1表示涨，0表示跌3,14,20,63,126
# for i in [1,2,3,4,5]:
#     X = data[[f'break{i}' for i in [3,14,20,63,126]]]  # 替换为你的特征列
#     y = data[f'expect{i}'].apply(lambda x: 1 if x>0 else 0)

#     # 拆分数据集为训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # 创建随机森林分类器
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

#     # 训练模型
#     rf_model.fit(X_train, y_train)

#     # 进行预测
#     y_pred = rf_model.predict(X_test)

#     # 输出模型性能
#     print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))

    

#     # 保存模型
#     joblib.dump(rf_model, f'data/RF_Data/random_forest_model{i}.pkl')

# # 加载模型
for i in [1,2,3,4,5]:
    loaded_model = joblib.load(f'data/RF_Data/random_forest_model{i}.pkl')
    y_pred=loaded_model.predict(pd.DataFrame([[1,1,1,1,1]],columns=[f'break{i}' for i in [3,14,20,63,126]]))
    print(y_pred[0])

# # 使用加载的模型进行预测
# y_pred_loaded = loaded_model.predict(X_test)