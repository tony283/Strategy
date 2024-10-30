import rqdatac
import pandas as pd
rqdatac.init()
from datetime import datetime
#price = rqdatac.futures.get_dominant_price("CU",start_date="20100104", end_date="20240809",frequency='1d',fields=None,adjust_type='post', adjust_method='prev_close_ratio')
#print(price)
#price.to_excel("data/raw_price_CU_daily.xlsx")
date=datetime.today()
daten=datetime.strptime()
def get_raw_data(name):
    price = rqdatac.futures.get_dominant_price(name,start_date="20100104",frequency='1d',fields=None,adjust_type='post', adjust_method='prev_close_ratio')
    print(price)
    price.to_excel(f"data/raw_price_{name}_daily.xlsx")
    multi =rqdatac.futures.get_contract_multiplier([name], start_date='20100104',end_date=""  market='cn')
    multi.to_excel(f"data/multi_{name}_daily.xlsx")
    print(multi)
def merge(name):
    a:pd.DataFrame = pd.read_excel(f"data/raw_price_{name}_daily.xlsx")
    b =pd.read_excel(f"data/multi_{name}_daily.xlsx")
    a["multiplier"]= b["contract_multiplier"]
    print(a)
    a["profit"]=(a["close"]-a["prev_close"])/a["prev_close"]
    for i in [5,20,40,63,126,252]:
        a[f'sigma{i}']=a["profit"].rolling(window=i).std()
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
    load(i)