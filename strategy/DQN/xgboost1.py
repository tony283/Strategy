import time
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import cupy as cp
import matplotlib.pyplot as plt
# Load dataset

v141=["break1","break3",'break14','break20','break63','break126','d_vol','d_oi','mmt_open','high_close','low_close','corr_price_vol','corr_price_oi','corr_ret_vol','corr_ret_oi','corr_ret_dvol','corr_ret_doi','norm_turn_std','vol_skew5','vol_skew14','vol_skew20','vol_skew63','vol_skew126','vol_skew252','price_skew5','price_skew14','price_skew20','price_skew63','price_skew126','price_skew252']
v142=["break1","break3",'break14','break20','break63','break126','d_vol','d_oi','high_close','low_close','corr_price_vol','corr_price_oi','corr_ret_oi','corr_ret_dvol','corr_ret_doi','norm_turn_std','vol_skew5','vol_skew14','vol_skew252','price_skew5','price_skew14','price_skew20','price_skew126']
for it in range(1,3):
    t=time.time()
    data = pd.read_excel('data/RF_Data/rf_old.xlsx')
    data = data.dropna()
    X = data[v142]  # Replace with your feature columns
    y = data[f'expect{it}'].apply(lambda x: 1 if x > 0 else 0)
    print(f'factor num: {len(X.columns)}')
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create XGBoost classifier with GPU support
    model = xgb.XGBClassifier(eval_metric='mlogloss', subsample=0.9,learning_rate=0.05)  # Use 'cuda' for GPU

    # Set the parameter grid
    param_grid = {
        'objective': ['binary:logistic'],  # For binary classification
        'n_estimators': [150,200,250],
        'max_depth': [4],
        'gamma': [0,0.05,0.1]
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                            scoring='f1', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    # Output the best parameters and score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # Use the best model for predictions
    best_model = grid_search.best_estimator_
    # Predict using the original test data
    y_pred = best_model.predict(X_test)

    # Output classification report
    print(classification_report(y_test, y_pred))
    print(time.time()-t)

    # Save the model
    # joblib.dump(best_model, f'data/RF_Data/XGBoost_v1_4_2_{it}.pkl')

# best:xgb.XGBClassifier = joblib.load(f'data/RF_Data/XGBoost_v1_4_1_{3}.pkl')
# # print(best)
# # plot
# plt.bar(["break1","break3",'break14','break20','break63','break126','d_vol','d_oi','mmt_open','high_close','low_close','corr_price_vol','corr_price_oi','corr_ret_vol','corr_ret_oi','corr_ret_dvol','corr_ret_doi','norm_turn_std','vol_skew5','vol_skew14','vol_skew20','vol_skew63','vol_skew126','vol_skew252','price_skew5','price_skew14','price_skew20','price_skew63','price_skew126','price_skew252'], best.feature_importances_)
# plt.show()