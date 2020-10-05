# -*- coding: utf-8 -*-
"""
Params.

sigunguCd: 시군구코드 - 11680
bjdongCd: 법정동코드 - 10300
bun: 번 - 0012
ji: 지 - 0000
useYm: 사용년월 - 201501
numOfRows
pageNo

@author: ISP
"""

import requests
import os
from requests import get 
import datetime
import time
import re
import json
import scipy.interpolate as spi
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from datetime import timedelta
import csv
import smtplib
from email.mime.text import MIMEText
import statistics

def Filtering_data(data_contents):
    index_value_s = data_contents.find("</sigunguCd><useQty>")
    index_value_e = data_contents.find("</useQty><useYm>")
    value = np.float64(data_contents[index_value_s+len("</sigunguCd><useQty>"):index_value_e])
    return(value)


#API input
service_key= "YOUR KEY"

sigunguCd = "11680"
bjdongCd = "10300"
bun = "0012"
ji = "0000"
use_Y = ["2017","2018","2019","2020"]
use_m = ["01","02","03","04","05","06","07","08","09","10","11","12"]

escape_Ym = "202006" # No data

#Data collection
data_list = []
for i in range(len(use_Y)):
    for j in range(len(use_m)):
        
        useYm = use_Y[i]+use_m[j]
        if useYm == escape_Ym:
            break
        
        url_power = "http://apis.data.go.kr/1611000/BldEngyService/getBeElctyUsgInfo?sigunguCd="+sigunguCd+"&bjdongCd="+bjdongCd+"&bun="+bun+"&ji="+ji+"&useYm="+useYm+"&ServiceKey="+service_key
        url_gas = "http://apis.data.go.kr/1611000/BldEngyService/getBeGasUsgInfo?sigunguCd="+sigunguCd+"&bjdongCd="+bjdongCd+"&bun="+bun+"&ji="+ji+"&useYm="+useYm+"&ServiceKey="+service_key
        
        #API request
        data_API_power = requests.get(url_power)
        data_API_gas = requests.get(url_gas)
        
        
        data_contents_power = data_API_power.text
        data_contents_gas = data_API_gas.text
    
        # list_data_contents = re.split("<category>", data_contents)

        value_power = Filtering_data(data_contents_power)
        value_gas = Filtering_data(data_contents_gas)
        
        data_list.append([use_Y[i], use_m[j], value_power, value_gas])


data_fin = pd.DataFrame(data_list)
data_fin.columns = ["Year","Month","Power","Gas"]
# data_fin.to_csv("filename.csv")
