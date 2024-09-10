import rqdatac
import pandas as pd
rqdatac.init()
#price = rqdatac.futures.get_dominant_price("CU",start_date="20100104", end_date="20240809",frequency='1d',fields=None,adjust_type='post', adjust_method='prev_close_ratio')
#print(price)
#price.to_excel("data/raw_price_CU_daily.xlsx")
def get_raw_data(name):
    price = rqdatac.futures.get_dominant_price(name,start_date="20100104", end_date="20240809",frequency='1d',fields=None,adjust_type='post', adjust_method='prev_close_ratio')
    print(price)
    price.to_excel(f"data/raw_price_{name}_daily.xlsx")
    multi =rqdatac.futures.get_contract_multiplier([name], start_date='20100104', end_date="20240809", market='cn')
    multi.to_excel(f"data/multi_{name}_daily.xlsx")
    print(multi)
def merge(name):
    a:pd.DataFrame = pd.read_excel(f"data/raw_price_{name}_daily.xlsx")
    b =pd.read_excel(f"data/multi_{name}_daily.xlsx")
    a["multiplier"]= b["contract_multiplier"]
    print(a)
    a["profit"]=(a["close"]-a["prev_close"])/a["prev_close"]
    a.to_excel(f"data/{name}_daily.xlsx")
    
def load(name):
    get_raw_data(name)
    merge(name)
    
    
typelist=['A']
for t in typelist:
    load(t)