import pandas as pd


file_name = "section/newsecbreak/"
real_name = "newsecbreak"
tail =""
#新建df
indexes :dict={"Range": ["0.15"]}
columns ={"N": [19]}
orginal_value =1e9


index_title = list(indexes.keys())[0]
column_title = list(columns.keys())[0]

for r in indexes[index_title]:
    for s in columns[column_title]:
        name= real_name+f"_{index_title}{r}_{column_title}{s}{tail}"#需修改
        plots = pd.read_excel("back/trade/"+"Trade"+real_name+f"_{index_title}{indexes[index_title][0]}_{column_title}{columns[column_title][0]}{tail}.xlsx")
        typelist = plots["type"].drop_duplicates()
        df = pd.DataFrame(columns=["Win_Rate","total","mean_profit"],index=typelist)
        time = plots["date"].drop_duplicates()
        
        for future_type in typelist:
            future_plot = plots[plots["type"]==future_type]
            buy_plot = future_plot[future_plot["B/S"]=="B"]
            sell_plot = future_plot[future_plot["B/S"]=="S"]
            loop = len(sell_plot)
            if loop==0:
                continue
            count=0
            sum_profit=0
            for i in range(loop):
                close = sell_plot["price"].iloc[i]
                prev_close = buy_plot["price"].iloc[i]
                direction = sell_plot["direction"].iloc[i]
                amount = sell_plot["amount"].iloc[i]
                sum_profit+= (close-prev_close)*amount if direction=="long" else -(close-prev_close)*amount
                
                d = 1 if direction=="long" else -1
                if d*(close-prev_close)>0:
                    count+=1
                
            win_rate = count/loop
            df.loc[future_type,"Win_Rate"]=win_rate
            df.loc[future_type,"total"]=loop
            df.loc[future_type,"mean_profit"]=sum_profit/loop
        df=df.sort_values(by="Win_Rate")
        df.to_excel(f"back/profitanalyzer/{real_name}win_rate_result.xlsx")
                
