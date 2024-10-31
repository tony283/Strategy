import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report
import pandas as pd
import joblib
# 加载数据集
data = pd.read_excel('data/RF_Data/rf.xlsx')
data=data.dropna()
X = data[[f'break{i}' for i in [3,14,20,63,126]]]  # 替换为你的特征列
y = data[f'expect{3}'].apply(lambda x: 1 if x>0 else 0)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 XGBoost 分类器
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# 设置参数范围
param_grid = {
    'objective': 'multi:softmax',  # 多分类问题
    'num_class': 2, 
    'eval_metric':'mlogloss',
    'n_estimators': [50, 100,150,200],
    'max_depth': [3, 5, 7,10],
    'learning_rate': [0.01, 0.1, 0.3],
    'eta':[0.01,0.05,0.1,0.3],
    'gamma':[0,0.1,0.2]
}

# 网格搜索
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 输出分类报告
print(classification_report(y_test, y_pred))
joblib.dump(best_model, f'data/RF_Data/XGBoost_v1_0_0_{1}.pkl')