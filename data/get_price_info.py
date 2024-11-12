import rqdatac
import pandas as pd
import numpy as np
rqdatac.init()
from datetime import datetime


#price = rqdatac.futures.get_dominant_price("CU",start_date="20100104", end_date="20240809",frequency='1d',fields=None,adjust_type='post', adjust_method='prev_close_ratio')
#print(price)
#price.to_excel("data/raw_price_CU_daily.xlsx")
date=datetime.today()
daten=datetime.strftime(date,"%Y%m%d")


def get_raw_data(name):
    price = rqdatac.futures.get_dominant_price(name,start_date="20100104",end_date=daten,frequency='1d',fields=None,adjust_type='post', adjust_method='prev_close_ratio')
    print(price)
    price.to_excel(f"data/raw_price_{name}_daily.xlsx")
    multi =rqdatac.futures.get_contract_multiplier([name], start_date='20100104',end_date=daten,  market='cn')
    multi.to_excel(f"data/multi_{name}_daily.xlsx")
    print(multi)
def merge(name):
    a:pd.DataFrame = pd.read_excel(f"data/raw_price_{name}_daily.xlsx")
    b =pd.read_excel(f"data/multi_{name}_daily.xlsx")
    a["multiplier"]= b["contract_multiplier"]
    a["profit"]=(a["close"]-a["prev_close"])/a["prev_close"]
    a['amount']=a['volume']*a['multiplier']
    a['high-low']=a['high']-a['low']
    a['prevclose-low']=a['prev_close']-a['low']
    a['high-prevclose']=a['high']-a['prev_close']
    for i in [5,20,40,63,126,252]:
        a[f'sigma{i}']=a["profit"].rolling(window=i).std()
    for i in [1,3,14,20,63,126,252]:
        a[f'break{i}']=(a["close"]-a["close"].shift(i))/(a["close"].shift(i)*np.sqrt(i)*a["sigma20"])
    for i in [1,2,3,4,5]:
        a[f'expect{i}']=(a["close"].shift(-i)-a["close"])/(a["close"])
    a['d_vol']=(a["volume"]-a["volume"].shift(i))/a["volume"].shift(1)
    a['d_oi']=(a["open_interest"]-a["open_interest"].shift(i))/a["open_interest"].shift(1)
    a['mmt_open']=(a["open"]-a["close"].shift(i))/a["close"].shift(1)#开盘动量
    a['high_close']=(a['high']-a['close'])/a['close']
    a['low_close']=(a['close']-a['low'])/a['close']
    a['corr_price_vol']=a['close'].rolling(window=20).corr(a['volume'])
    a['corr_price_oi']=a['close'].rolling(window=20).corr(a['open_interest'])
    a['corr_ret_vol']=a['profit'].rolling(window=20).corr(a['volume'])
    a['corr_ret_oi']=a['profit'].rolling(window=20).corr(a['open_interest'])
    a['corr_ret_dvol']=a['profit'].rolling(window=20).corr(a['d_vol'])
    a['corr_ret_doi']=a['profit'].rolling(window=20).corr(a['d_oi'])
    a['turnover']=a['close']*a['volume']
    a['sigma_turnover']=a['turnover'].rolling(window=20).std()
    a['ave_turnover']=a['turnover'].rolling(window=20).mean()
    a['norm_turn_std']=a['sigma_turnover']/a['ave_turnover']
    
    for i in [5,14,20,63,126,252]:
        a[f'vol_skew{i}']=a['volume'].rolling(window=i).skew()
    for i in [5,14,20,63,126,252]:
        a[f'price_skew{i}']=a['close'].rolling(window=i).skew()
    for i in [5,14,20,63,126,252]:
        a[f'sigma_skew{i}']=a['sigma5'].rolling(window=i).skew()
    
    a['low_close_high']=(2*a['close']-a['high']-a['low'])/(a['high']-a['low'])
    a['d_low_close_high']=a['low_close_high']-a['low_close_high'].shift()
    a['mean6']=a['close']/a['close'].rolling(window=6).mean()-1
    a['mean12']=a['close']/a['close'].rolling(window=12).mean()-1
    a['dif']=a['close'].ewm(span=12,adjust=False).mean()-a['close'].ewm(span=26,adjust=False).mean()
    a['dea']=a['dif'].ewm(span=9,adjust=False).mean()
    a['macd']=2*(a['dif']-a['dea'])
    a['sma_low_close_high9']=(a['close']-a['low'].rolling(window=9).min())/(a['high'].rolling(window=9).max()-a['low'].rolling(window=9).min()).ewm(com=3).mean()
    a['sma_low_close_high6']=(a['close']-a['low'].rolling(window=6).min())/(a['high'].rolling(window=6).max()-a['low'].rolling(window=6).min()).ewm(com=20).mean()
    a['std_vol6']=a['volume'].rolling(window=6).std()/a['volume'].rolling(window=6).mean()
    a['ddif_vol']=(a['close'].rolling(window=9).mean()-a['close'].rolling(window=26).mean())/a['close'].rolling(window=12).mean()
    a['norm_ATR']=a[['high-prevclose','prevclose-low','high-low']].max(axis=1)/a['close']
    a['sq5_low_close_open_high']=(a['close']-a['low'])*(a['open']/a['close']).apply(lambda x:x**5)/(a['close']-a['high'])
    for i in [5,14,20,63,126,252]:
        a[f'vol_kurt{i}']=a['volume'].rolling(window=i).kurt()
    for i in [5,14,20,63,126,252]:
        a[f'price_kurt{i}']=a['close'].rolling(window=i).kurt()
    for i in [5,14,20,63,126,252]:
        a[f'sigma_kurt{i}']=a['sigma5'].rolling(window=i).kurt()
    for i in [5,20,63,126]:
        a[f'winrate{i}']=a['profit'].rolling(window=i).apply(lambda x:np.sum(x>0))/i
    for i in [5,20,63,126]:
        a[f'draw{i}']=(a['close'].rolling(window=i).max()-a['close'].rolling(window=i).min())/a['close'].rolling(window=i).max()
    for i in [5,20,63,126]:
        a[f'position{i}']=(a['close']-a['close'].rolling(window=i).min())/(a['close'].rolling(window=i).max()-a['close'].rolling(window=i).min())
    a['d_position5']=a['position5']-a['position20']
    a['d_position20']=a['position20']-a['position63']
    a['d_position63']=a['position63']-a['position126']
    for i in [5,20]:
        a[f'daily_position{i}']=((a['close']-a['low'])/(a['high']-a['low'])).ewm(span=i).mean()
    a['d_daily_position']=a['daily_position20']-a['daily_position5']
    ##非指标
    a['amihud']=a['profit'].apply(abs)/(a['amount']*a['close'])
    for i in [5,20,63,126]:
        a[f'relative_amihud{i}']=a['amihud'].ewm(span=i).mean()/a['amihud'].rolling(window=20).mean()
        a[f'highlow_avg{i}']=(a['high']/a['low']-1).rolling(window=i).mean()
        a[f'highlow_std{i}']=(a['high']/a['low']).rolling(window=i).std()
        a[f'upshadow_avg{i}']=(1-a[['open','close']].max(axis=1)/a['high']).rolling(window=i).mean()
        a[f'upshadow_std{i}']=(1-a[['open','close']].max(axis=1)/a['high']).rolling(window=i).std()
        a[f'downshadow_avg{i}']=(a[['open','close']].min(axis=1)/a['low']-1).rolling(window=i).mean()
        a[f'upshadow_std{i}']=(a[['open','close']].min(axis=1)/a['low']-1).rolling(window=i).std()
    a['high_m_low']=a['high']*a['low']
    a['MAX(close-SMA(close,5))']=pd.concat([a['close'].ewm(span=5,adjust=False).mean(),a['close']],axis=1).max(axis=1)
    print(a)
    a.to_excel(f"data/{name}_daily.xlsx")
    a.to_csv(f"data/{name}_daily.csv")
    
def load(name):
    get_raw_data(name)
    merge(name)
    
tl=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']    
for i in tl:
    print(i)
    load(i)

