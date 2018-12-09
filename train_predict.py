from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pickle
import numpy as np
import time
import pandas as pd

train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')

print('Training...')
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 多分类的问题
    'gamma': 0.1,                 # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,               # 构建树的深度，越大越容易过拟合
    'subsample': 0.8,             # 随机采样训练样本
    'colsample_bytree': 0.8,      # 生成树时进行的列采样
    'min_child_weight': 2,
    'silent': 1,                  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,                   # 如同学习率
    'alpha':1e-5,
    'lambda':1e-5,
    'seed': 1000,
    'scale_pos_weight':1,
    'nthread': 4,                  # cpu 线程数
}

dtrain = xgb.DMatrix(train_data, label=train_label,)
num_round = 200
bst = xgb.train(params, dtrain, num_round)

print('Predicting...')
dtest = xgb.DMatrix(test_data,)
pred = bst.predict(dtest)

print('Generating CSV file...')
submit = pd.DataFrame({'UID':test_id,'Tag':pred})
submit = submit.set_index('UID',drop=True)
time_str = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
submit.to_csv('submit_{}.csv'.format(time_str))