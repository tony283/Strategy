import pandas as pd


file_name = "section/newsec/"
real_name = "newsec"
tail =""
#新建df
indexes :dict={"R":[14]}
columns ={"H": [2]}
orginal_value =1e9


index_title = list(indexes.keys())[0]
column_title = list(columns.keys())[0]

for r in indexes[index_title]:
    for s in columns[column_title]:
        name= real_name+f"_{index_title}{r}_{column_title}{s}{tail}"#需修改
        
        plots = pd.read_excel("back/trade/"+"Trade"+real_name+f"_{index_title}{indexes[index_title][0]}_{column_title}{columns[column_title][0]}{tail}.xlsx")
        typelist = plots["type"].drop_duplicates()
        df = pd.DataFrame(columns=["mean_profit"],index=typelist)
        time = plots["date"].drop_duplicates()
        
        for future_type in typelist:
            future_plot = plots[plots["type"]==future_type]
            buy_plot = future_plot[future_plot["B/S"]=="B"]
            sell_plot = future_plot[future_plot["B/S"]=="S"]
            loop = len(sell_plot)
            if loop==0:
                continue
            mean_profit =0
            for i in range(loop):
                close = sell_plot["price"].iloc[i]
                prev_close = buy_plot["price"].iloc[i]
                if sell_plot["direction"].iloc[i]=="long":
                    mean_profit+= (close-prev_close)/prev_close
                else:
                    mean_profit += (prev_close-close)/prev_close
            mean_profit/=loop
            df.loc[future_type,"mean_profit"]=mean_profit
            
        df=df.sort_values(by="mean_profit")
        df.to_excel("back/result.xlsx")
                
