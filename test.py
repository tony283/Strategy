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
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
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

#     s = df["sigma20"].iloc[-1]
#     breaklist=[(df["close"].iloc[-1]-df["close"].iloc[-R-1])/(s*df["close"].iloc[-R-1]*np.sqrt(R)) for R in [3,14,20,63,126]]
#     la.append(breaklist)
# X=np.array(la).T
# corr=X@X.T
# print(corr)
# eigenvalue, featurevector = np.linalg.eig(corr)

# print("特征值：", eigenvalue)
# print("特征向量：", featurevector)
# a=featurevector[np.argmax(eigenvalue)]
# print(np.sign(a.sum()))
a=np.array([1,1])
b=np.array([[1,1]])
print(a)
print(b)
print(a@a.T)
print(b@b.T)