import numpy as np
import pandas as pd

'''
数据预处理
xgboos里不用标准化和缺失数据处理
'''
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape)
print(test.shape)

# delete the row with all nan in train
train.dropna(axis=0,how='all',inplace=True)

train_num = train.shape[0]
test_num = test.shape[0]

drop_list = ['Id']
# delete the column with all nan
for c in test.columns:
    if train[c].isnull().values.all() or test[c].isnull().values.all():
        drop_list.append(c)

# 如果一列有很多缺失值(>0.9),那么这一列没有多大的意义，不如丢弃(可以替代上面全nan的丢弃方法）
for c in test.columns:
    # print(train[c].isnull().sum())
    if train[c].isnull().sum() > train_num*0.9:
        drop_list.append(c)

print('this columns should be delete from data, since too many nan')
print(len(set(drop_list)),set(drop_list))

train.drop(columns=drop_list,inplace=True)
test.drop(columns=drop_list,inplace=True)

# 分离训练集data和label
train_label = train['label']
train.drop(columns='label',inplace=True)

print(train.shape)
print(test.shape)

# 将训练集和测试集合并，方便onehot处理
data = pd.concat([train,test],axis=0)
print(data.shape)

# # get all data_type, here is {dtype('int64'), dtype('float64'), dtype('O')}
# data_type = []
# for key in data.columns:
#     data_type.append(data[key].dtype)
# data_type = set(data_type)

# # 用均值填补nan,xgboost里不需要
# for c in data.columns:
#     # object列要特殊处理
#     if data[c].dtype != np.dtype('object'):
#         mean = data[c][:train_num].mean()
#         data[c].fillna(mean,inplace=True)
#         assert data[c].isnull().sum() == 0

one_hot_columns = []
object_columns = []
for key in data.columns:
    if data[key].dtype == np.dtype('object'):
        object_columns.append(key)
    elif data[key].dtype == np.dtype('int64') and data[key].max()-data[key].min()<10 and data[key].max()-data[key].min()>1:
        one_hot_columns.append(key)
print('object columns:',object_columns)
print('one-hot columns :',one_hot_columns)

# # normalize,对于不需要onehot的列标准化,xgboost里不需要
# for c in data.columns:
#     if c not in one_hot_columns and c not in object_columns:
#         dmean = data[c][:train_num].mean()
#         dstd = data[c][:train_num].std()
#         data[c] = (data[c] - dmean) / dstd

# 处理object列
# In_165:日期，只提取年月
data['In_165_year'] = data['In_165'].apply(lambda x:int(x[:4]))
data['In_165_month'] = data['In_165'].apply(lambda x:int(x[5:7]))
data.drop(columns='In_165',inplace=True)
# In_166:字符串,型号？
data['In_166_type1'] = data['In_166'].apply(lambda x:x[:2])
data['In_166_type2'] = data['In_166'].apply(lambda x:x[-2:])
data['In_166_int'] = data['In_166'].apply(lambda x:int(x[2:-2]))
data.drop(columns='In_166',inplace=True)

# 把type1和type2列加入需要onthot的列中
one_hot_columns.extend(['In_166_type1','In_166_type2'])

# 将值域较小的int64列和['In_166_type1','In_166_type2'] ont-hot编码
for key in one_hot_columns:
    append_column = pd.get_dummies(data[key])
    append_column = append_column.rename(columns = lambda x:key+'_'+str(x))   #改名,避免列名冲突
    data = pd.concat([data,append_column],axis=1)
data.drop(columns=one_hot_columns,inplace=True)

# # 另一种手动onehot的方法
# from sklearn import model_selection, preprocessing
# for col in cat_cols:
#     print(col)
#     lbl = preprocessing.LabelEncoder()
#     lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
#     train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
#     test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

data = np.array(data)
train_data = data[:train_num]
test_data = data[train_num:]

print('Data Shape & Type:')
print('train_data:',train_data.shape,train_data.dtype)
print('train_label:',train_label.shape,train_label.dtype)
print('test_data:',test_data.shape,test_data.dtype)

np.save('train_data.npy',train_data)
np.save('train_label.npy',train_label)
np.save('test_data.npy',test_data)
