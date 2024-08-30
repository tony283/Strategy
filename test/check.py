import pandas as pd
import numpy as np
data_list = ["BU","M"]
for future_type in data_list:
    data = pd.read_excel(f"data/{future_type}_daily.xlsx")
    back = pd.read_excel(f"test/Tradevolbottom_S40_R0.1.xlsx")
    data = data[data["date"]>=back["date"].iloc[0]]
    data = data[data["date"]<=back["date"].iloc[-1]]
    data = data[["date","close"]]
    back = back[back["type"]==future_type]
    data["B"] = np.nan
    data["S"] = np.nan
    data = data.set_index("date")
    print(data)
    for index,row in back.iterrows():
        print(row["date"])
        if row["B/S"]=="B":
            data.loc[row["date"],"B"] = row["price"]
        elif row["B/S"]=="S":
            data.loc[row["date"],"S"] = row["price"]
    print(data)
    data.to_excel(f"test/{future_type}_result.xlsx")