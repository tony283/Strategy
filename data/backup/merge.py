import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
fmt= "%Y-%m-%d"
a:pd.DataFrame = pd.read_excel("data/raw_price_CU_daily.xlsx")
b =pd.read_excel("data/multi_CU_daily.xlsx")
a["multiplier"]= b["contract_multiplier"]
print(a)
a.to_excel("data/CU_daily.xlsx")