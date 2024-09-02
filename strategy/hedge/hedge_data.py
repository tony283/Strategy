import pandas as pd
from datetime import datetime

typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']

# trading_dates = pd.read_excel("data/trading_dates.xlsx")
# trading_dates = trading_dates[trading_dates['date']<datetime(2024,8,30)]
# dates = trading_dates["date"].iloc[0]
# df = pd.DataFrame(index=trading_dates["date"],columns=typelist)
# # print(df)

# for i in typelist:
#     temp = pd.read_excel(f"data/{i}_daily.xlsx",index_col ="date")
#     for index, item in temp.iterrows():
#         if( index<dates):
#             continue
#         df.loc[index,i] =item["profit"]

# df.to_excel("factor/future_profit_data.xlsx")


df = pd.read_excel("factor/future_profit_data.xlsx",index_col="date")
print(df)
corre = df.corr()
corre.to_excel("factor/future_corre_data.xlsx")
