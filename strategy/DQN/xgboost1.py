import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# Load dataset
# data = pd.read_excel('data/RF_Data/rf.xlsx')
# data = data.dropna()
# X = data[[f'break{i}' for i in [3, 14, 20, 63, 126]]]  # Replace with your feature columns
# y = data[f'expect{3}'].apply(lambda x: 1 if x > 0 else 0)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create XGBoost classifier with GPU support
# model = xgb.XGBClassifier(eval_metric='mlogloss', device='cuda')  # Use 'cuda' for GPU

# # Set the parameter grid
# param_grid = {
#     'objective': ['binary:logistic'],  # For binary classification
#     'n_estimators': [50, 100, 150, 200],
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'gamma': [0, 0.1, 0.2],
# }

# # Perform grid search
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
#                            scoring='accuracy', cv=5, verbose=1)
# grid_search.fit(X_train, y_train)

# # Output the best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# # Use the best model for predictions
# best_model = grid_search.best_estimator_
# # Predict using the original test data
# y_pred = best_model.predict(X_test)

# # Output classification report
# print(classification_report(y_test, y_pred))

# # Save the model
# joblib.dump(best_model, f'data/RF_Data/XGBoost_v1_0_0_{3}.pkl')

best:xgb.XGBClassifier = joblib.load(f'data/RF_Data/XGBoost_v1_0_0_{3}.pkl')
print(best.predict(pd.DataFrame([[-1,1,1,1,1]],columns=[f'break{i}' for i in [3, 14, 20, 63, 126]])))
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
# plot
plot_tree(best,fmap='', num_trees=0, rankdir='UT', ax=None)
plt.show()