'''
:@Author: LRF
:@Date: 11/5/2024, 3:33:23 PM
:@LastEditors: LRF
:@LastEditTime: 11/5/2024, 3:33:23 PM
:Description: 
'''
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing.pool
import pandas as pd
import numpy as np
import requests
from strategy.utils.utils import *
from datetime import datetime
from dateutil import rrule
import matplotlib.pyplot as plt
import random
import multiprocessing
import pandas as pd
import time

def generate_data(future):
    data =pd.read_excel(f"data/{future}_daily.xlsx",index_col=0)
    #
    data["sigma"]=data["profit"].rolling(window=63).std()
    data["profit20"]=data["profit"].rolling(window=20).sum()/(np.sqrt(20)*data["sigma"])
    data["profit63"]=data["profit"].rolling(window=63).sum()/(np.sqrt(63)*data["sigma"])
    data["profit126"]=data["profit"].rolling(window=126).sum()/(np.sqrt(126)*data["sigma"])
    data["profit252"]=data["profit"].rolling(window=252).sum()/(np.sqrt(252)*data["sigma"])
    data["close_sigma"]=data["close"].rolling(window=63).std()
    data["macd1"]=(data["close"].ewm(span=8,adjust=False).mean()-data["close"].ewm(span=24,adjust=False).mean())/data["close_sigma"]
    data["macd2"]=(data["close"].ewm(span=16,adjust=False).mean()-data["close"].ewm(span=48,adjust=False).mean())/data["close_sigma"]
    data["macd3"]=(data["close"].ewm(span=32,adjust=False).mean()-data["close"].ewm(span=96,adjust=False).mean())/data["close_sigma"]
    data["profit"] = data["profit"]/data["sigma"]
    data["expect"] = data["profit"].shift(-1)


    data=data[["date","profit","profit20","profit63","profit126","profit252","macd1","macd2","macd3","expect"]]
    # data=pd.DataFrame({"A":[1,2,3,4,5,6,7,8,9],"B":[2,3,1,3,43,1,32,13,4]})
    data=data.dropna()
    print(data)
    print(data.columns)
    data.to_excel(f"strategy/lstm/train_data/{future}.xlsx")
    
typelist = ['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']


# threshold=0.015
# vol_list=[]
# for i in typelist:
#     a=pd.read_excel(f"data/backup/{i}_daily.xlsx")
#     a["profit"]=a['profit'].apply(abs)
# # print(a["close"].to_numpy())
# # fft= np.fft.fft(a["close"].to_numpy(),252)
# # x=pd.DataFrame(np.abs(fft))
# # print(x)
# # x.loc[1:].plot()
# # plt.show()
#     a=a[["profit","sigma20"]].corr()
#     if a.loc["profit","sigma20"]<threshold:
#         continue
#     vol_list.append(i)
# print(vol_list)
# from statsmodels.tsa.seasonal import STL
# plt.rc("figure", figsize=(10, 6))
 
# df=pd.read_excel("data/A_daily.xlsx")[["date","close"]]
# df['date']=pd.to_datetime(df['date'])
# df.set_index('date',inplace=True)
# print(df)
# res = STL(df,period=252).fit()
# res.plot()
# plt.show()
# df['trend']=res.trend
# df['seasonal']=res.seasonal
# df['resid']=res.resid
# 替换为你的邮箱和密码
# import imaplib
# import email
# from email.header import decode_header
# EMAIL = '370318641@qq.com'
# PASSWORD = 'fgkzhjgvtakubiha'
# IMAP_SERVER = 'imap.qq.com'

# # 连接到邮箱
# mail = imaplib.IMAP4_SSL(IMAP_SERVER)
# mail.login(EMAIL, PASSWORD)

# # 选择收件箱
# mail.select('inbox')
# pattern="商品收益互换成交信息20241016"
# # 搜索邮件（可以根据需要修改搜索条件）
# status, messages = mail.search(None, f'SUBJECT "{pattern}"'.encode('utf-8'))
# mail_ids = messages[0].split()

# # 取最新一封邮件
# latest_email_id = mail_ids[-1]

# # 获取邮件
# status, msg_data = mail.fetch(latest_email_id, '(RFC822)')
# msg = email.message_from_bytes(msg_data[0][1])

# # 获取邮件主题
# subject, encoding = decode_header(msg['Subject'])[0]
# if isinstance(subject, bytes):
#     subject = subject.decode(encoding if encoding else 'utf-8')

# print(f'Subject: {subject}')

# # 获取邮件正文
# if msg.is_multipart():
#     for part in msg.walk():
#         if part.get_content_type() == 'text/html':
#             html_content = part.get_payload(decode=True).decode()
#             break
# else:
#     html_content = msg.get_payload(decode=True).decode()
# print(html_content)
# la=[]
# for i in typelist:
#     df =pd.read_excel(f"data/{i}_daily.xlsx",index_col=0)
#     s = df["sigma20"].iloc[-1]
#     breaklist=[(df["close"].iloc[-1]-df["close"].iloc[-R-1])/(s*df["close"].iloc[-R-1]*np.sqrt(R)) for R in [3,14,20,63,126]]
#     print(breaklist)
#     la.append(breaklist)
# X=np.array(la).T
# corr=X@X.T/len(la)
# print(corr)
# eigenvalue, featurevector = np.linalg.eig(corr)

# print("特征值：", eigenvalue)
# print("特征向量：", featurevector)
# a=featurevector[np.argmax(eigenvalue)]

# p=[]
# df= pd.DataFrame(columns=[f'break{i}' for i in [3,14,20,63,126]]+[f'expect{i}' for i in [1,2,3,4,5]])
# for future_type in typelist:
#     m_data=pd.read_excel(f"data/{future_type}_daily.xlsx")
#     m_data=m_data[m_data["date"]<datetime(2017,12,1)]
#     print(m_data)
#     m_data=m_data[["break1","break3",'break14','break20','break63','break126','d_vol','d_oi','mmt_open','high_close','low_close','corr_price_vol','corr_price_oi','corr_ret_vol','corr_ret_oi','corr_ret_dvol','corr_ret_doi','norm_turn_std','vol_skew5','vol_skew14','vol_skew20','vol_skew63','vol_skew126','vol_skew252','price_skew5','price_skew14','price_skew20','price_skew63','price_skew126','price_skew252','low_close_high','d_low_close_high','mean6','mean12','dif','dea','macd','sma_low_close_high9','sma_low_close_high6','std_vol6','ddif_vol','norm_ATR','sq5_low_close_open_high','expect1','expect2','expect3','expect4','expect5']]
#     if len(df)==0:
#         df=m_data
#     else:
#         df=pd.concat([df,m_data])

# df=df.replace([np.inf, -np.inf], np.nan).dropna()
# print(df)
# df.to_excel("data/RF_Data/rf_old.xlsx",index=False)

factors=['sigma5', 'sigma20', 'sigma40', 'sigma63', 'sigma126', 'sigma252', 'break1', 'break3', 'break14', 'break20', 'break63', 'break126', 'break252','d_vol', 'd_oi', 'mmt_open', 'high_close', 'low_close', 'corr_price_vol', 'corr_price_oi', 'corr_ret_vol', 'corr_ret_oi', 'corr_ret_dvol', 'corr_ret_doi', 'turnover', 'sigma_turnover', 'ave_turnover', 'norm_turn_std', 'vol_skew5', 'vol_skew14', 'vol_skew20', 'vol_skew63', 'vol_skew126', 'vol_skew252', 'price_skew5', 'price_skew14', 'price_skew20', 'price_skew63', 'price_skew126', 'price_skew252', 'sigma_skew5', 'sigma_skew14', 'sigma_skew20', 'sigma_skew63', 'sigma_skew126', 'sigma_skew252', 'low_close_high', 'd_low_close_high', 'mean6', 'mean12', 'dif', 'dea', 'macd', 'sma_low_close_high9', 'sma_low_close_high6', 'std_vol6', 'ddif_vol', 'norm_ATR', 'sq5_low_close_open_high', 'vol_kurt5', 'vol_kurt14', 'vol_kurt20', 'vol_kurt63', 'vol_kurt126', 'vol_kurt252', 'price_kurt5', 'price_kurt14', 'price_kurt20', 'price_kurt63', 'price_kurt126', 'price_kurt252', 'sigma_kurt5', 'sigma_kurt14', 'sigma_kurt20', 'sigma_kurt63', 'sigma_kurt126', 'sigma_kurt252', 'winrate5', 'winrate20', 'winrate63', 'winrate126', 'draw5', 'draw20', 'draw63', 'draw126', 'position5', 'position20', 'position63', 'position126', 'd_position5', 'd_position20', 'd_position63', 'daily_position5', 'daily_position20', 'd_daily_position', 'relative_amihud5', 'highlow_avg5', 'highlow_std5', 'upshadow_avg5', 'upshadow_std5', 'downshadow_avg5', 'relative_amihud20', 'highlow_avg20', 'highlow_std20', 'upshadow_avg20', 'upshadow_std20', 'downshadow_avg20', 'relative_amihud63', 'highlow_avg63', 'highlow_std63', 'upshadow_avg63', 'upshadow_std63', 'downshadow_avg63', 'relative_amihud126', 'highlow_avg126', 'highlow_std126', 'upshadow_avg126', 'upshadow_std126', 'downshadow_avg126']
# factors.extend(['expect1','expect2','expect3','expect4','expect5'])
# import rqdatac
# import pandas as pd
# import numpy as np
# corr=pd.DataFrame()
# for i in typelist:
#     a=pd.read_excel(f'data/{i}_daily.xlsx')
#     a=a[a["date"]>datetime(2018,1,1)][factors]
#     if len(corr)==0:
#         corr=a.corr()
#     else:
#         corr=corr+a.corr()
# corr=corr/len(typelist)
# corr=corr[['expect1','expect2','expect3','expect4','expect5']].iloc[:-5]
# corr.to_excel('factor/factor_exposure.xlsx')

a=pd.read_csv("data/CU_daily.csv")

print(a['vol_skew20'].apply(lambda x: max(x,1)).rolling(12).std()-a['vol_skew20'].apply(lambda x: max(x,1)).rolling(12).std().shift())
b=a['vol_skew20'].apply(lambda x: max(x,1)).rolling(12).std()-a['vol_skew20'].apply(lambda x: max(x,1)).rolling(12).std().shift()
print(b.mean())