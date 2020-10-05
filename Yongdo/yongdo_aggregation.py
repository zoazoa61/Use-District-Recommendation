# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 01:25:39 2020

@author: ISP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%%  Input
path = 'K:/33. 대구 빅데이터/2. 분석/Analysis_final_201003/'
filename = 'new_data_중구_bus병합.csv'
data = pd.read_csv(path + filename, dtype={'시군구코드': str, '법정동코드': str, '번': str, '지': str},
                   encoding='euc-kr')


#%%    
"1. 용도지역 aggregation"
idx_different = np.zeros([len(data),1])
for i in range(len(idx_different)-1):
    if data.loc[i, 'jijiguCdNm'] != data.loc[i+1, 'jijiguCdNm']:
        idx_different[i] = 0
    else: 
        idx_different[i] = 1

idx_different_location = np.where(idx_different==0)[0]

#%% gosi_year~승하차수 까지만 유의미하게 쓸 수 있음
aggregation_list = []
idx_start = 0
for j in range(len(idx_different_location)):
    idx_end = idx_different_location[j]+1
    
    tmp_v = data[idx_start:idx_end].mean()#.transpose()
    
    template = data[idx_start:idx_start+1]
    
    for i, l in enumerate(tmp_v.index):
        if l in template.columns:
            template[l] = tmp_v[l]
    aggregation_list.append(template)   
    idx_start = idx_end
        
#%%
data_agg = aggregation_list[0]
for i in range(1, len(aggregation_list)):
    data_agg = pd.concat([data_agg, aggregation_list[i]])
data_agg = data_agg.reset_index(drop=True)

data_agg.to_csv("aggregation_"+filename, index=None, encoding='euc-kr')
