from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
with open('best_param.pkl','rb') as f:
    param = pickle.load(f)

# load param
model = XGBClassifier()
model.set_params(**param)
# train
model.fit(train_data,train_label)

# show feature importance
print(model.feature_importances_)
print(len(model.feature_importances_))
print(sum(model.feature_importances_))
xgb.plot_importance(model)
pyplot.show()

# predict
pre = model.predict_proba(test_data)[:,1]
# print('AUC:',roc_auc_score(train_label,pre))
print(pre)