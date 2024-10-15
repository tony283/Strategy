import rqdatac
import pandas as pd

typelist=['AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']

a=pd.read_excel("future_total.xlsx",index_col=0)
b=a.rolling(window=20,).std()
print(b)

b.to_excel("data/future_std.xlsx")


