import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import RobustScaler

#%%  Input: data 지정 => bus / building
# path = 'D:/33. 대구 빅데이터/2. 분석/Analysis_final/data/'
path = 'D:/33. 대구 빅데이터/2. 분석/Analysis_final_201003/'
file_name = 'concat_aggregation_new_data_total.csv'
data = pd.read_csv(path + file_name, 
                   dtype={'시군구코드': str, '법정동코드': str, '번': str, '지': str}, 
                   encoding='euc-kr')

yongdo_v, yongdo_c= np.unique(data['jijiguCdNm'].values, return_counts = True)

data.rename(columns={"승하차수":"usage_bus"}, inplace=True)
data.rename(columns={"정류소여부":"busstop_exist"}, inplace=True)

#%% nan processing
data_col_name = data.columns
col_x = np.concatenate([np.arange(8,21)])
# col_x = np.concatenate([np.arange(8,11),np.arange(16,18),np.arange(20,21)])
# col_x = np.concatenate([np.arange(8,11),np.arange(16,18)])

data = data[data_col_name[col_x]]


#%% nan processing
data.dropna(inplace = True, axis='rows')
data = data.reset_index(drop=True)

label = data['jijiguCdNm'].copy()# make label
data.drop(columns = ['jijiguCdNm'], inplace = True)


#%% feature, class 지정
features = data.columns
n_features = features.shape[0]
class_type = np.unique(label)
data = pd.DataFrame(RobustScaler().fit_transform(data)) # Roubustness for outliers

#%% corr
corrs = np.zeros((n_features, len(class_type))) # row: feature, col: 용도구역
for i in range(n_features):
    for j, c in enumerate(class_type):
        corrs[i,j] = np.corrcoef((label == c).astype(int), data.iloc[:,i])[1,0]

#%% importance score
importances = np.zeros((n_features, len(class_type))) # row: feature, col: 용도구역
import lightgbm as lgb
for j, c in enumerate(class_type):
    dtrain = lgb.Dataset(data, label=(label == c).astype(int).values)
    param = {'metric': 'l1', 'n_jobs': -1, 'colsample_bytree': 1.0, 'learning_rate': 0.01,
             'verbose':-1}
    model = lgb.train(param, train_set=dtrain ,num_boost_round=1000)
    y_pred = model.predict(data.values)
    importances[:,j] = model.feature_importance()

#%% plot
import seaborn as sns
# correlation
plt.figure(figsize=(10,5))
sns.heatmap(corrs.T)
plt.xticks(range(n_features), features, rotation=30)
plt.title('Correlation')
plt.show()
plt.show()

# importance score
plt.figure(figsize=(10,5))
for i in range(len(class_type)):
    plt.plot(importances[:,i], label=i)
plt.xticks(range(n_features), features, rotation=30)
plt.title('Feature importance score')
plt.legend()
plt.show()

#%% Save

corrs = pd.DataFrame(corrs)
importances = pd.DataFrame(importances)
label_yongdo = pd.DataFrame(class_type)
# corrs.to_csv("corr_"+file_name, header = None, index = None, encoding = 'euc-kr')
# importances.to_csv("importance_"+file_name, header = None, index = None, encoding = 'euc-kr')
# label_yongdo.to_csv("label_yongdo"+file_name, header = None, index = None, encoding = 'euc-kr')



