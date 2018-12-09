from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

'''
搜索xgboost的最佳参数
'''

# 绘图：模型在训练集和测试集上交叉验证的表现
def show_res(res):

    plot_range = range(res.shape[0])
    plt.plot(plot_range, res['train-auc-mean'], color='blue', label='training accuracy')
    plt.fill_between(plot_range, res['train-auc-mean'] + res['train-auc-std'],
                     res['train-auc-mean'] - res['train-auc-std'], alpha=0.15, color='blue')

    plt.plot(plot_range, res['test-auc-mean'], color='red', label='testing accuracy')
    plt.fill_between(plot_range, res['test-auc-mean'] + res['test-auc-std'],
                     res['test-auc-mean'] - res['test-auc-std'], alpha=0.15, color='red')

    plt.show()

train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')

params = {
    # 'tree_method': 'gpu_hist',
    # 'gpu_id': 1,
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 5,
    'subsample': 0.8,              # 随机采样训练样本
    'colsample_bytree': 0.8,       # 生成树时进行的列采样
    'min_child_weight': 1,
    'silent': 1,
    'eta': 0.5,
    'alpha':1e-3,
    'lambda':1e-3,
    'seed': 1000,
    'scale_pos_weight':1,
    'nthread': 4,
}

# 这里利用交叉验证和early_stop来确定最佳num_boost_round,也就是n_estimators(也就是模型内生成多少棵树,每一次boost都会生成一棵树)
# 注意，cv一般用来调参,所以不会保留生成的模型,这里需要把参数设置回原来的model里
# callback是用来边训练边显示进度的

# https://blog.csdn.net/ruding/article/details/78328835
# https://zhuanlan.zhihu.com/p/25308120
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

# 1.利用cv确定大致的迭代次数
dtrain = xgb.DMatrix(train_data,label=train_label)
res = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5,metrics='auc', seed=0,early_stopping_rounds=10,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
# 这里res.shape[0]就是early_stop时候的boost次数,这里稍微设置大一点
boost_num = res.shape[0]/0.9

# 2.利用gridsearchcv搜索其余参数,最好利用几轮搜索,来确定最优参数
# xgb为了兼容sklean的GridSearchCV借口,定义了模型类,参数基本一直,但是有三个参数的命名不同
params['n_estimators'] = boost_num
params['learning_rate'] = params['eta']
params['reg_alpha'] = params['alpha']
params['reg_lambda'] = params['lambda']
params.pop('eta')
params.pop('alpha')
params.pop('lambda')

# 设置想要搜索的参数及阈值
search_param = {
    'max_depth':[3,5],
    'min_child_weight':[2,4]
}
# 删除param里想要搜索的参数
for key in search_param:
    params.pop(key)

model = XGBClassifier()
model.set_params(**params)
gridsearch = GridSearchCV(estimator=model,param_grid=search_param,scoring='roc_auc',n_jobs=4,cv=5,iid=False)
gridsearch.fit(train_data,train_label)
print(gridsearch.cv_results_)
print(gridsearch.best_params_)
print(gridsearch.best_score_)

with open('best_param.pkl','wb') as f:
    pickle.dump(gridsearch.best_params_,f)


