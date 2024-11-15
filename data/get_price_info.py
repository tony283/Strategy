import rqdatac
import pandas as pd
import numpy as np
rqdatac.init()
import sys
sys.path.append("c:\\Users\\ROG\\Desktop\\Strategy\\strategy\\utils")
from datetime import datetime
import auto_utils

#price = rqdatac.futures.get_dominant_price("CU",start_date="20100104", end_date="20240809",frequency='1d',fields=None,adjust_type='post', adjust_method='prev_close_ratio')
#print(price)
#price.to_excel("data/raw_price_CU_daily.xlsx")
date=datetime.today()
daten=datetime.strftime(date,"%Y%m%d")


def get_raw_data(name):
    price = rqdatac.futures.get_dominant_price(name,start_date="20100104",end_date=daten,frequency='1d',fields=None,adjust_type='post', adjust_method='prev_close_ratio')
    price.to_excel(f"data/raw_price_{name}_daily.xlsx")
    multi =rqdatac.futures.get_contract_multiplier([name], start_date='20100104',end_date=daten,  market='cn')
    multi.to_excel(f"data/multi_{name}_daily.xlsx")
def merge(name):
    f=auto_utils.FunctionPool()
    a:pd.DataFrame = pd.read_excel(f"data/raw_price_{name}_daily.xlsx")
    b =pd.read_excel(f"data/multi_{name}_daily.xlsx")
    a["multiplier"]= b["contract_multiplier"]
    a["profit"]=(a["close"]-a["prev_close"])/a["prev_close"]
    a['amount']=a['volume']*a['multiplier']
    a['vol_std']=a['volume'].rolling(window=20).std()
    a['oi_std']=a['open_interest'].rolling(window=20).std()
    a['price_std']=a['close'].rolling(window=20).std()
    
    
    for i in [5,20,40,63,126,252]:
        a[f'sigma{i}']=a["profit"].rolling(window=i).std()
    for i in [1,3,14,20,63,126,252]:
        a[f'break{i}']=(a["close"]-a["close"].shift(i))/(a["close"].shift(i)*np.sqrt(i)*a["sigma20"])
    for i in [1,2,3,4,5]:
        a[f'expect{i}']=(a["close"].shift(-i)-a["close"])/(a["close"])
    a['high-low']=(a['high']-a['low'])/a['price_std']
    a['prevclose-low']=(a['prev_close']-a['low'])/a['price_std']
    a['high-prevclose']=(a['high']-a['prev_close'])/a['price_std']
    a['d_vol']=(a["volume"]-a["volume"].shift(1))/(a['vol_std'])
    a['d_oi']=(a["open_interest"]-a["open_interest"].shift(1))/(a['oi_std'])
    a['mmt_open']=(a["open"]-a["close"].shift(1))/(a["close"].shift(1)*a['sigma20'])#开盘动量
    a['high_close']=(a['high']-a['close'])/(a['close']*a['sigma20'])-1
    a['low_close']=(a['close']-a['low'])/(a['close']*a['sigma20'])-1
    a['corr_price_vol']=2*a['close'].rolling(window=20).corr(a['volume'])/0.4
    a['corr_price_oi']=a['close'].rolling(window=20).corr(a['open_interest'])/0.4
    a['corr_ret_vol']=a['profit'].rolling(window=20).corr(a['volume'])/0.4
    a['corr_ret_oi']=a['profit'].rolling(window=20).corr(a['open_interest'])/0.3
    a['corr_ret_dvol']=a['profit'].rolling(window=20).corr(a['d_vol'])/0.3
    a['corr_ret_doi']=a['profit'].rolling(window=20).corr(a['d_oi'])/0.3
    a['turnover']=a['close']*a['volume']
    a['sigma_turnover']=a['turnover'].rolling(window=20).std()
    a['ave_turnover']=a['turnover'].rolling(window=20).mean()
    a['norm_turn_std']=a['sigma_turnover']/(a['ave_turnover']*0.2)-2
    
    for i in [5,14,20,63,126,252]:
        a[f'vol_skew{i}']=a['volume'].rolling(window=i).skew()-1.5
    for i in [5,14,20,63,126,252]:
        a[f'price_skew{i}']=a['close'].rolling(window=i).skew()
    for i in [5,14,20,63,126,252]:
        a[f'sigma_skew{i}']=a['sigma5'].rolling(window=i).skew()-1.5
    
    a['low_close_high']=2*(2*a['close']-a['high']-a['low'])/(a['high']-a['low'])
    a['d_low_close_high']=(a['low_close_high']-a['low_close_high'].shift())*0.5
    a['mean6']=(a['close']/a['close'].rolling(window=6).mean()-1)/a['sigma20']
    a['mean12']=(a['close']/a['close'].rolling(window=12).mean()-1)/a['sigma20']
    a['dif']=(a['close'].ewm(span=12,adjust=False).mean()-a['close'].ewm(span=26,adjust=False).mean())/(a['price_std']/2)
    a['dea']=a['dif'].ewm(span=9,adjust=False).mean()
    a['macd']=2*(a['dif']-a['dea'])
    a['sma_low_close_high9']=3*(a['close']-a['low'].rolling(window=9).min())/(a['high'].rolling(window=9).max()-a['low'].rolling(window=9).min()).ewm(com=3).mean()-1.5
    a['sma_low_close_high6']=3*(a['close']-a['low'].rolling(window=6).min())/(a['high'].rolling(window=6).max()-a['low'].rolling(window=6).min()).ewm(com=20).mean()-1.5
    a['std_vol6']=a['volume'].rolling(window=6).std()/a['vol_std']-1
    a['norm_ATR']=a[['high-prevclose','prevclose-low','high-low']].max(axis=1)-1
    a['sq5_low_close_open_high']=(a['close']-a['low'])*(a['open']/a['close']).apply(lambda x:x**5)/(a['close']-a['high'])
    for i in [5,14,20,63,126,252]:
        a[f'vol_kurt{i}']=a['volume'].rolling(window=i).kurt()
        a[f'price_kurt{i}']=a['close'].rolling(window=i).kurt()
        a[f'sigma_kurt{i}']=a['sigma5'].rolling(window=i).kurt()
    for i in [5,20,63,126]:
        a[f'winrate{i}']=9*(a['profit'].rolling(window=i).apply(lambda x:np.sum(x>0))/i-0.5)
        a[f'position{i}']=3*((a['close']-a['close'].rolling(window=i).min())/(a['close'].rolling(window=i).max()-a['close'].rolling(window=i).min())-0.5)
    a['d_position5']=a['position5']-a['position20']
    a['d_position20']=a['position20']-a['position63']
    a['d_position63']=a['position63']-a['position126']
    for i in [5,20]:
        a[f'daily_position{i}']=9*(((a['close']-a['low'])/(a['high']-a['low'])).ewm(span=i).mean()-0.5)
    a['d_daily_position']=a['daily_position20']-a['daily_position5']
    ##非指标
    a['amihud']=a['profit'].apply(abs)/(a['amount']*a['close'])
    a['amihud_std']=a['amihud'].rolling(window=20).std()
    for i in [5,20,63,126]:
        a[f'relative_amihud{i}']=(a['amihud'].ewm(span=i).mean()-a['amihud'].rolling(window=20).mean())/a['amihud_std']
    a['high_m_low']=a['high']*a['low']
    a['MAX(close-SMA(close,5))']=pd.concat([a['close'].ewm(span=5,adjust=False).mean(),a['close']],axis=1).max(axis=1)
    a['d_position']=a['position5']-a['position5'].shift()
    for i in [5,20,63,126]:
        a[f'skew_position{i}']=a[f'position{i}'].rolling(window=10).skew()
    a['sigma_skew20_m_position63']=a['sigma_skew20']*a['position63']
    a['sigma_skew20_m_d_position5']=-a['sigma_skew20']*a['d_position5']
    a['ADD[d_position5 , PROD[vol_skew126 , skew_position63]]']=a['d_position5']+a['vol_skew126']*a['skew_position63']
    a['DIF5(skew_position63)']=-a['skew_position63']+a['skew_position63'].shift(5)
    a['RANK9(skew_position63)']=-a['skew_position63'].rolling(window=9).rank()
    a['ADD[skew_position20 , position63]']=-a['skew_position20']-a['position63']
    a['PROD[RANK26(vol_kurt126) , low_close]']=-a['vol_skew126'].rolling(window=26).rank()*a['low_close']
    a['MINUS[skew_position63 , relative_amihud5]']=-a['skew_position63']+a['relative_amihud5']
    a['MINUS[high_close , skew_position63]']=a['high_close']-a['skew_position63']
    a['ADD[vol_skew126 , skew_position63]']=-a['vol_skew126']-a['skew_position63']
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
