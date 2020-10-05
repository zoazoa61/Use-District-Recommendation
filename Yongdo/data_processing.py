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
# data 지정 => bus / building
path = 'K:/33. 대구 빅데이터/2. 분석/Analysis_final_201003/버스+건물/'
filename = 'data_고산동_bus병합.csv'
data = pd.read_csv(path + filename, dtype={'시군구코드': str, '법정동코드': str, '번': str, '지': str}, encoding='utf-8')

#%% Input data columns
# data = data.loc[:,['jiga', 'jijiguCdNm', 'Power', 'Gas', 'bcRat', 'vlRat', 'hhldCnt', 'fmlyCnt','grndFlrCnt','정류소여부','승하차수']]
data_col_name = data.columns
col_x = np.concatenate([np.arange(0,16),np.arange(17,18),np.arange(19,21),np.arange(22,24)])
data = data[data_col_name[col_x]]
#%% 전처리1: 구역, 지구 삭제
for i, l in enumerate(data['jijiguCdNm'].values):
    if type(l) != str:
        continue
    #문자 끝에 2개가 지역
    if l[-2:] != '지역':
        data.loc[i, 'jijiguCdNm'] = np.nan
#%% 전처리2: 용도지역을 큰 단위로 통합 및 용도지역 외 데이터 제거
yongdo_name = pd.unique(data['jijiguCdNm'])
for i, l in enumerate(data['jijiguCdNm'].values):
    #큰단위로 통합
    if l in ['제1종일반주거지역', '제2종일반주거지역', '제3종일반주거지역']:
        data.loc[i, 'jijiguCdNm'] = '일반주거지역'
    if l in ['제2종전용주거지역']:
        data.loc[i, 'jijiguCdNm'] = '전용주거지역'
    #용도지역 외 데이터 제거
    # if l in [yongdo_name[9],yongdo_name[13],yongdo_name[14]]:# for 중구
    # for 고산동
    if l in [yongdo_name[8]]: 
    # if l in [yongdo_name[9],yongdo_name[7]]:# for 신당동
        data = data.drop(i)
        
data = data.reset_index(drop=True)
#yongdo_name_processed = pd.unique(data['jijiguCdNm'])#check after preprocessing

#%%    
"1. 용도지역 interpolation"
# data.loc[0, 'jijiguCdNm'] = "일반상업지역" #for 중구
# data.loc[0, 'jijiguCdNm'] = "일반상업지역" #for 고산동
# data.loc[0, 'jijiguCdNm'] = "일반주거지역" #for 신당동

idx_nan_jijiguCdNm = pd.isna(data['jijiguCdNm'])

save_jijiguCdNm = data['jijiguCdNm'][0]
for i in range(len(idx_nan_jijiguCdNm)):
    if idx_nan_jijiguCdNm.values[i]== True:
        data.loc[i, 'jijiguCdNm'] = save_jijiguCdNm
    else:
        save_jijiguCdNm = data['jijiguCdNm'][i]

#%%
"2. 지가 nan 값은 날림(대부분 데이터 없음)"
idx_nan_jiga = np.isnan(data['jiga'][:]) #지가에 nan값 있는지 확인 후 인덱스 마킹

for i in range(len(idx_nan_jiga)):
    if idx_nan_jiga.values[i]== True:
        data = data.drop(i)
data = data.reset_index(drop=True)
#%%
"3. 0값의 Power interpolation"
idx_zero_power = data['Power'] == 0
count = 0
for i in range(len(idx_zero_power)):
    if idx_zero_power.values[i]== False:
        break
    count +=1
save_power = data['Power'][count]

for i in range(len(idx_zero_power)):
    if idx_zero_power.values[i]== True:
        data.loc[i, 'Power'] = save_power
    else:
        save_power = data['Power'][i]

"3-1. 0값의 Gas interpolation"

idx_zero_gas = data['Gas'] == 0
count = 0

for i in range(len(idx_zero_gas)):
    if idx_zero_gas.values[i]== False:
        break
    count +=1
save_gas = data['Gas'][count]

for i in range(len(idx_zero_gas)):
    if idx_zero_power.values[i]== True:
        data.loc[i, 'Gas'] = save_gas
    else:
        save_gas = data['Gas'][i]

#%%
"4. Drop na"
data = data.dropna()
data = data.reset_index(drop=True)

data.to_csv("new_"+filename, index=None, encoding='euc-kr')
