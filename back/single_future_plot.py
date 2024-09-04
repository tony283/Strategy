import pandas as pd
import numpy as np
future_type = "LC"
file_path = "newsec_R14_H1"



df =pd.read_excel(f"data/{future_type}_daily.xlsx")
df=df.set_index("date")
df=df["close"]
print(df)
df["long_buy"]=np.nan
df["long_sell"]=np.nan
df["short_buy"]=np.nan
df["short_sell"]=np.nan
future_df = pd.read_excel(f"back/trade/Trade{file_path}.xlsx")
future_df = future_df[future_df["type"]==future_type]
future_long = future_df[future_df["direction"]=="long"]
future_short = future_df[future_df["direction"]=="short"]

print(future_long.index)
for index,value in future_long.iterrows():
    if value["B/S"]=="B":
        print(value["date"])
        df.loc[value["date"],"long_buy"]=value["price"]
    else:
        df.loc[value["date"],"long_sell"]=value["price"]
for index,value in future_short.iterrows():
    if value["B/S"]=="B":
        df.loc[value["date"],"short_buy"]=value["price"]
    else:
        df.loc[value["date"],"short_sell"]=value["price"]

df.to_excel(f"Report/{future_type}_report.xlsx")