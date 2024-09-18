import pandas as pd


file_name = "section/newsecbreakall/"
real_name = "newsecbreakall"
tail =""
#新建df
indexes :dict={"R": ["19"]}
columns ={"H": [2]}
orginal_value =1e9


index_title = list(indexes.keys())[0]
column_title = list(columns.keys())[0]
for r in indexes[index_title]:
    for s in columns[column_title]:
        name= real_name+f"_{index_title}{r}_{column_title}{s}{tail}"#需修改
        plots = pd.read_excel("back/trade/"+"Trade"+real_name+f"_{index_title}{indexes[index_title][0]}_{column_title}{columns[column_title][0]}{tail}.xlsx")
        typelist = plots["type"].drop_duplicates()
        df = pd.DataFrame(columns=["Win_Rate","total","mean_profit","equal_profit"],index=typelist)
        time = plots["date"].drop_duplicates().copy().to_frame()
        time["av_cash"] = None
        b_plot = plots[plots["B/S"]=="B"]
        for index,row in time.iterrows():
            sec = b_plot[b_plot["date"]==row["date"]].copy()
            sec["cash"] = sec["amount"]*sec["price"]
            av_cash =sec["cash"].mean()
            time.loc[index,"av_cash"]=av_cash
        time=time.set_index("date")
        for future_type in typelist:
            future_plot = plots[plots["type"]==future_type]
            buy_plot = future_plot[future_plot["B/S"]=="B"]
            sell_plot = future_plot[future_plot["B/S"]=="S"]
            loop = len(sell_plot)
            if loop==0:
                continue
            count=0
            sum_profit=0
            equal_profit =0
            for i in range(loop):
                close = sell_plot["price"].iloc[i]
                prev_close = buy_plot["price"].iloc[i]
                direction = sell_plot["direction"].iloc[i]
                amount = sell_plot["amount"].iloc[i]
                av=time.loc[buy_plot["date"].iloc[i],"av_cash"]
                weight = prev_close*amount/av
                sum_profit+= (close-prev_close)*weight/prev_close if direction=="long" else -(close-prev_close)*weight/prev_close
                equal_profit +=(close-prev_close)/prev_close if direction=="long" else -(close-prev_close)/prev_close
                d = 1 if direction=="long" else -1
                if d*(close-prev_close)>0:
                    count+=1
                
            win_rate = count/loop
            df.loc[future_type,"Win_Rate"]=win_rate
            df.loc[future_type,"total"]=loop
            df.loc[future_type,"mean_profit"]=sum_profit/loop
            df.loc[future_type,"equal_profit"]=equal_profit/loop
        df=df.sort_values(by="Win_Rate")
        df.to_excel(f"back/profitanalyzer/{real_name}win_rate_result.xlsx")
                
