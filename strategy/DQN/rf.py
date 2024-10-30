import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import glob
# 读取数据
data = pd.read_excel('data/RF_Data/rf.xlsx')
data=data.dropna()

# 其中 'target' 是1表示涨，0表示跌3,14,20,63,126
def GenerateRF(X,y):
    X = data[[f'break{i}' for i in [3,14,20,63,126]]]  # 替换为你的特征列
    y = data[f'expect{i}'].apply(lambda x: 1 if x>0 else 0)

    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
    'n_estimators': [10,15,20,25,30],
    'max_depth': [3,4,5,6],
    'min_samples_split': [10,20,40],
    'min_samples_leaf': [5,10,15],
    'max_features': ['auto', 'sqrt', 'log2']
    }
    # 创建随机森林分类器
    print('No.1:')
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print("最佳参数:", grid_search.best_params_)
    # 训练模型
    best_rf = grid_search.best_estimator_
    # 进行预测
    y_pred = best_rf.predict(X_test)

    # 输出模型性能
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    

    # 保存模型
    joblib.dump(best_rf, f'data/RF_Data/random_forest_model_v1_0_4_{i}.pkl')

# # 加载模型
# for i in [1,2,3,4,5]:
#     loaded_model = joblib.load(f'data/RF_Data/random_forest_model{i}.pkl')
#     y_pred=loaded_model.predict(pd.DataFrame([[1,1,1,1,1]],columns=[f'break{i}' for i in [3,14,20,63,126]]))
#     print(y_pred[0])

# # 使用加载的模型进行预测
# y_pred_loaded = loaded_model.predict(X_test)

# loaded_model = joblib.load(f'data/RF_Data/random_forest_model_v1_0_3_{5}.pkl')
# print(loaded_model.get_params())

# a=glob.glob("data/RF_data/*.pkl")

# model = joblib.load(f'data/RF_Data/random_forest_model_v1_0_2_{4}.pkl')

# print(model.predict(pd.DataFrame([[0,5,0,1,1]],columns=[f'break{i}' for i in [3,14,20,63,126]])))
    
    






