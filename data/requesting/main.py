import requestdata
import pandas as pd
import numpy as np
M=3
R=14
N=20
H=2
RANGE=0.15
L=7
REALCASH = 1000000
USINGRATE=0.5

CASH=REALCASH*10*USINGRATE



BookTable = pd.DataFrame(columns=["future_type","total_turnover","volume","direction","price"])
typelist = ['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
# requestdata.request_all()
m_data = requestdata.request_old_data()
current = m_data["CU"]["date"].iloc[-1]
print(f"current_date: {current}")
multiplier = pd.read_excel("data/multiplier.xlsx",index_col=0)
temp_dict =[]#用于储存收益率信息
for future_type in typelist:
    try:
        profit = (m_data[future_type]["close"].iloc[-1]-m_data[future_type]["close"].iloc[-1-R])/m_data[future_type]["close"].iloc[-1-R]
        maxindex = max(0,len(m_data[future_type])-1-M)
        profitM = (m_data[future_type]["close"].iloc[-1]-m_data[future_type]["close"].iloc[maxindex])/m_data[future_type]["close"].iloc[maxindex]
        sigma = m_data[future_type]["profit"].iloc[-N:].std()
        liquidity = m_data[future_type].iloc[-L:].copy()
        liquidity["liquidity"] = np.log(liquidity["profit"].apply(lambda x:abs(x))/(liquidity["close"]*liquidity["volume"]/1e11)+1)
        liquidity = liquidity["liquidity"].mean()
        temp_dict.append([future_type,profit,sigma,profitM,liquidity])
    except:
        continue
ranking = pd.DataFrame(temp_dict,columns=["future_type","profit","sigma","profitM","liquidity"])
ranking = ranking[ranking["sigma"]!=0]
ranking = ranking.dropna()
ranking["break"] = ranking["profitM"].apply(lambda x:abs(x)/np.sqrt(M))*ranking["liquidity"]/ranking['sigma']
ranking = ranking[ranking["break"]!=0]
range=int(RANGE*len(ranking))
ranking = ranking.sort_values(by="profit",ascending=True)#排名
cash_max = (CASH//(2))
highest = ranking.iloc[-range:]["break"].sum()
lowest = ranking.iloc[:range]["break"].sum()
for index, row in ranking.iloc[-range:].iterrows():#收益率最高的
    future_type=row["future_type"]
    proportion = row["break"]/highest
    close = m_data[future_type]["close"].iloc[-1]
    multi = multiplier.loc[future_type,"multiplier"]
    
    buy_amount = int(cash_max*proportion/(close*multi))
    if buy_amount<=0:
        continue
    BookTable.loc[len(BookTable)]=[future_type,cash_max*proportion,int(cash_max*proportion/(close*multi)),"long",close]
for index, row in ranking.iloc[:range].iterrows():#收益率最低的
    future_type=row["future_type"]
    proportion = row["break"]/lowest
    close = m_data[future_type]["close"].iloc[-1]
    multi = multiplier.loc[future_type,"multiplier"]
    buy_amount = int(cash_max*proportion/(close*multi))
    if(buy_amount<=0):
        continue
    BookTable.loc[len(BookTable)]=[future_type,cash_max*proportion,int(cash_max*proportion/(close*multi)),"short",close]
    
print(BookTable)