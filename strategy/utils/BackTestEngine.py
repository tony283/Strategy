
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import time
import multiprocessing
# @profile
def try_set_value(a:dict,key,value,is_close=True,close=0):
    """_summary_

    Args:
        a (dict): _description_
        key (_type_): _description_
        value (_type_): [amount,price,margin]
    """
    assert value[0]>=0
    if (key in a.keys()):
        a[key][1]*=a[key][0]
        value[1]*=value[0]
        a[key]+=value

        a[key][1]/=a[key][0]
        a[key] = a[key].astype(np.int64)
        t4=time.time()

    else:
        a[key]=np.array(value,dtype=np.int64)
    if not is_close:
        info = key.split("_")
        earn = (close-value[1])*value[0] if info[1]=="long" else (value[1]-close)*value[0]
        a[key][2]+=earn
def try_sell_value(a:dict,key,value,direction,is_close=True,close=0):
    """_summary_

    Args:
        a (dict): _description_
        key (_type_): _description_
        value (_type_): [amount,price]
        direction (_type_): _description_
    """
    assert value[0]>=0
    assert direction=="long" or direction=="short"
    real_key = key+"_"+direction
    assert value[0]<=a[real_key][0]
    earn=0
    
    if real_key not in a.keys():
        return 0
    if is_close:
        earn=  int(a[real_key][2])*int(value[0])//int(a[real_key][0])
        a[real_key][2] -=earn#amount*close_price
        a[real_key][0] -=value[0]

    else:
        earn=  int(a[real_key][2])*int(value[0])//int(a[real_key][0])
        a[real_key][2] -=earn#amount*close_price
        a[real_key][0] -=value[0]
        real_earn = (value[1]-close)*value[0] if direction=="long" else -(value[1]-close)*value[0]
        earn +=real_earn
    return earn
    
    
    
    



class context():
    def __init__(self) -> None:
        pass
class ReadingError(Exception):
    pass    
def timer(func):
    def func_wrapper(*args,**kwargs):
        from time import time
        time_start = time()
        result = func(*args,**kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper
   

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):  
    """  
    Call in a loop to create terminal progress bar  
    @params:  
        iteration   - Required  : current iteration (Int)  
        total       - Required  : total iterations (Int)  
        prefix      - Optional  : prefix string (Str)  
        suffix      - Optional  : suffix string (Str)  
        decimals    - Optional  : positive number of decimals in percent complete (Int)  
        length      - Optional  : character length of bar (Int)  
        fill        - Optional  : bar fill character (Str)  
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)  
    """  
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))  
    filledLength = int(length * iteration // total)  
    bar = fill * filledLength + '-' * (length - filledLength)  
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)  
    # Print New Line on Complete  
    if iteration == total:   
        print()  


class BackTest():
    """
    context: 可以自由定义数据
    Position
            ├── cash(int)当前拥有的现金(单位0.01分)
            │
            ├── original_cash(int)初始现金(单位0.01分)
            │
            ├── asset(int)当前资产包括现金的价值(单位0.01分)
            │
            └── hold(dict)当前所有持仓信息/
                │
                ├── key(str)持仓品种，格式为"CU_long","CU_short",long,short代表多头或者空头
                │   
                └── value(np.ndarray)当前品种的持仓信息，格式为np.array([仓位(int)(已算上合约乘数),加权平均成本*10000(int),保证金余量*10000(int)])
    instrument(dict)保证金信息：{
                                "margin_rate":保证金比例(float),
                                "margin_limit":追加保证金标准
                                }
    data(dict)当前所有品种信息/
                │
                ├── key(str)持仓品种，格式为"CU"
                │   
                └── value(dataframe)当前品种的历史数据    
    """
    def __init__(self,cash=100000000,margin_rate=0.2,margin_limit=0.8,debug=True):
        """_summary_

        Args:
            cash (int, optional): 初始资金，默认100000000.
            margin_rate (float, optional): 保证金比例. Defaults to 0.2.
            margin_limit (float, optional): 保证金下限，当保证金低于保证金下限*期货价值*保证金比例时触发补交保证金. Defaults to 0.8.
            debug (bool, optional): debug为True时会打印每笔交易以及debug信息，正常情况下直接设为false加快回测速度. Defaults to True.
        """
        self.context=context()#context存储了一切想要存储的内容，比如事先准备好的factor，或者参数，可以在后面的before_trade，after_trade，和handle_bar引用
        self.debug = debug
        self.data = {}#这里将会存储所有已定阅品种的历史数据，订阅在subscribe中进行
        self.position = context()#position记录了当前持仓，现金等信息，具体信息看最上面的注释
        # 由于浮点数计算有精度误差，对于大浮点数后几位会出现精度丢失的情况，比如2e8 + 100还是2e8，
        # 因此一切计算尽量都使用int类型，将财产乘10000，以0.01分作为最小单元
        self.position.cash=cash*10000
        self.position.original_cash = cash*10000#参考基准，不会变
        self.position.hold = {}#hold item格式：[amount,price*10000,guarantee]，具体见最上面注释
        self.position.asset=[]#每日的投资组合公允价值（含现金）最终都会被追加到此
        
        #self.hold =pd.DataFrame(columns=["date","type","hold","direction","average_cost"])#历史持仓，太耗时，已取消
        self.trade_record = pd.DataFrame(columns=["date","type","amount","direction","B/S","price"])#成交记录，最终会生成对账单
        self.temp_trade=[]
        self.instrument = {"margin_rate":margin_rate,"margin_limit":margin_limit}#设置保证金比例和下限
        self.init(self.context)#自定义初始化

        
        pass
    def init(self,context):
        """_summary_

        Args:
            context (_type_): 自定义初始化的信息都放在这里，注意context.name必须设置，它会作为最终回测曲线xlsx的文件名
        """
        pass
    def subscribe(self,future_type:str):
        """_summary_

        Args:
            future_type (str): 订阅 品种加载到self.data

        
        """
        try:
            future_data = pd.read_excel("data/"+future_type+"_daily.xlsx")

        except:
            raise ReadingError("Cannot load data/"+future_type+"_daily.xlsx")
        
        self.data[future_type]=future_data
    def _subscribe(self,future_type:str):
        """_summary_

        Args:
            future_type (str): 订阅 品种加载到self.data

        
        """
        try:
            future_data = pd.read_excel("data/"+future_type+"_daily.xlsx")

        except:
            raise ReadingError("Cannot load data/"+future_type+"_daily.xlsx")
        
        return future_data
    def subscribe_parallel(self,typelist:list):

        typelist=typelist

        with ProcessPoolExecutor(20) as executor: # 创建 ThreadPoolExecutor 
            future_list = executor.map(self._subscribe, [file for file in typelist]) # 提交任务

        
        for future in future_list:
            self.data[future[0]]=future[1]

        print(self.data["CU"])

    
    def Rating(self,m_data,*args)->pd.DataFrame:
        rate=pd.DataFrame({"type":self.context.typelist,"rate":[0]*len(self.context.typelist)})
        
        for func in args:
            ranking = func(m_data)
            rate=pd.merge(rate,ranking,on="type",how="inner")
        for i in rate.columns:
            if i=="type" or i=="rate":
                rate["rate"]=rate["rate"]+rate[i]
        return rate
            
            
    
    
    
    
    def log(self,s:str):
        """_summary_

        debug为true时才会打印
        """
        if (not self.debug):
            pass
        else:
            print(s)
        
    def before_trade(self, context, m_data:pd.DataFrame):
        """_summary_

        需要重载
        """
        pass
    def open_handle(self, context, m_data:pd.DataFrame):
        """_summary_

        需要重载，适用于在开盘买入
        """
        pass
    def after_trade(self,context,):
        """_summary_

        需要重载
        """
        pass
    def handle_bar(self,m_data:pd.DataFrame,context):
        """_summary_

        Args:
            m_data (Dataframe): 这是到当前日为止的所有数据，注意回测的时候只能用最后一行的收盘价作为买入价，其余数据计算应当从倒数第二行开始，防止未来函数
            context (_type_): _description_
        """
        pass
    
    def process(self,m_data):
        """_summary_
        每日流程图 交易前干点事->盘位计算需要补交的保证金从现金扣除->在盘尾决定是否买卖->盘后做点事

        Args:
            m_data (_type_): 字典形式，例如m_data["CU"]是一个dataframe，保存了铜从历史最早的数据到当前交易日的所有数据
        """
        
        self.check_hold(m_data)#补交保证金
        self.handle_bar(m_data,self.context)
        self.after_trade(self.context)
    def check_hold(self,m_data):
        """_summary_
        补交保证金，在不加杠杆的条件下（margin_rate=1,margin_limit=0）不会生效

        Args:
            m_data (_type_): 同process
        """
        for future_type, value in m_data.items():
            try:
                current_close = value["close"].iloc[-1]
            except:
                continue
            if (future_type+"_long") in self.position.hold.keys():
                rest = self.position.hold[future_type+"_long"][2]#余量
                limit = int(self.position.hold[future_type+"_long"][0]*int(current_close*10000)*self.instrument["margin_rate"]*self.instrument["margin_limit"])#保证金下限
                if(rest<limit):
                    adding=int(int(current_close*10000)*(self.position.hold[future_type+"_long"][0]*self.instrument["margin_rate"]))-rest
                    self.position.cash-=adding
                    self.log(f"adding is {adding},rest is {rest}, limit is {limit}")
                    if(self.position.cash<0):
                        warnings.warn("保证金不足，补缴失败")
                    self.position.hold[future_type+"_long"][2]+=adding#补交保证金
            if (future_type+"_short") in self.position.hold.keys():
                rest = self.position.hold[future_type+"_short"][2]#余量
                limit = int(self.position.hold[future_type+"_short"][0]*int(current_close*10000)*self.instrument["margin_rate"]*self.instrument["margin_limit"])#保证金下限
                if(rest<limit):
                    adding=int(int(current_close*10000)*(self.position.hold[future_type+"_short"][0]*self.instrument["margin_rate"]))-rest
                    self.position.cash-=adding
                    if(self.position.cash<0):
                        warnings.warn("保证金不足，补缴失败")
                    self.position.hold[future_type+"_short"][2]+=adding#补交保证金
    
    def order_target_num(self,price,amount:int,multiplier:int,future_type:str,direction):
        """_summary_
        用于在handle_bar下单

        Args:
            price (_type_): 下单价
            amount (int): 下单手数
            multiplier (int): 合约乘数
            future_type (str): 期货种类代码
            direction (_type_): 方向 long or short 
        """
        assert direction=="long" or direction=="short"
        margin =int(price*10000*multiplier*amount*self.instrument["margin_rate"])
        if(margin>self.position.cash):
            warnings.warn("保证金不足，无法下单")
            return
        self.log(f"Bid-close:{price},order:{amount*multiplier}")
        try_set_value(self.position.hold,future_type+"_"+direction,np.array([amount*multiplier,
                                                                                 int(price*10000),
                                                                                 margin],dtype=np.int64
                                                                                )
                          )#如果下单失败直接会报错，但是对于无杠杆策略，已经在前面检测过不会失败
        self.position.cash-=margin
        self.temp_trade.append([self.current,future_type,amount*multiplier,direction,"B",price])
        return
    def sell_target_num(self,price,amount:int,multiplier:int,future_type:str,direction,is_close=True,close=0):
        """_summary_
        用于平仓

        Args:
            price (_type_): 卖单价
            amount (int): 手数
            multiplier (int): 合约乘数
            future_type (str): 期货代码
            direction (_type_): 方向
        """
        assert direction=="long" or direction=="short"
        if(amount<=0):
            warnings.warn(f"{self.current}:卖单量为0，无法完成交易，交易品种{future_type},交易数量{amount}")
            return
        if(True):
            self.log(f"ofr-close:{price},amount:{amount*multiplier}")
            earn=try_sell_value(self.position.hold,future_type,
                           np.array([amount*multiplier,int(price*10000)],dtype=np.int64),
                           direction,is_close,int(close*10000)
                          )
            self.position.cash+=earn
            self.temp_trade.append([self.current,future_type,amount*multiplier,direction,"S",price])
            # self.trade_record=pd.concat([self.trade_record,pd.DataFrame([[self.current,future_type,amount*multiplier,direction,"S",price]],
                                                                        # columns=self.trade_record.columns)],axis=0)
        return
    def sell_all_target(self, m_data):
        for future_type_dir, amount in self.position.hold.items():
                info = future_type_dir.split("_")
                future_type = info[0]
                direction = info[1]
                multi = m_data[future_type]["multiplier"].iloc[-1]
                if(amount[0]//multi<=0):
                    continue
                self.sell_target_num(m_data[future_type]["close"].iloc[-1],amount[0]//multi,multi,future_type,direction)
    #回测主函数
    # @timer
    def loop_process(self,start,end,saving_dir="back/"):
        """_summary_
        回测主函数

        Args:
            start (_type_): 开始日期，如 20120106
            end (_type_): 结束日期， 如20210101
        """
        time_series:pd.DataFrame = self.data[list(self.data.keys())[0]]["date"]
        time_series=time_series[time_series<=datetime.strptime(end,"%Y%m%d")]
        start_date = datetime.strptime(start,"%Y%m%d")
        real_time_series =time_series[time_series>=start_date]
        progress_total = len(real_time_series)
        current_date=time_series[time_series>=start_date].iloc[0]
        m_data={}#m_data拥有到今日的最新信息，位置在-1，但是今天用的历史数据不能有未来函数，策略用到的数据必须从-2往回
        for future_type, value in self.data.items():
            m_data[future_type] = value[value["date"]<current_date]#初始化start日期前的数据
        for i in range(len(real_time_series)):
            current_date=real_time_series.iloc[i]
            self.current=current_date
            for future_type, value in self.data.items():
                m_data[future_type]= value[value["date"]<=current_date]#每天先把当天数据导入，将close作为可以买卖的价格
            self.before_trade(self.context,m_data)
            self.open_handle(self.context,m_data)
            self.calculate_profit(m_data)#计算当日收益（分别计算每个品种看涨看跌的收益，将当日价格减去昨日价格）
            self.process(m_data)#进行买卖操作
            if(i%200==0):
                print_progress_bar(i + 1, progress_total, prefix=f'{self.context.name}:', suffix='Complete', length=20)
            #self.write_log(current_date)
        self.log(self.trade_record)
        real_time_series:pd.DataFrame=real_time_series.to_frame()
        real_time_series[self.context.name]=np.array(self.position.asset)
        real_time_series[self.context.name]=real_time_series[self.context.name].apply(lambda x:x/10000)
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        real_time_series.to_excel(saving_dir+"Back_"+self.context.name+".xlsx")
        print(f"Save completed! {saving_dir}Back_{self.context.name}.xlsx")
        #self.draw(self.context,real_time_series)
        #self.draw(self.context,real_time_series)
        #self.beautiful_plot()
        #self.statistics(real_time_series)
        self.trade_record =self.trade_record = pd.DataFrame(self.temp_trade,columns=["date","type","amount","direction","B/S","price"])
        self.trade_record.to_excel("back/trade/Trade"+self.context.name+".xlsx")
    
    def draw(self,context,df:pd.DataFrame):
        df.plot("date",self.context.name)
        plt.show()
    
    # @timer    
    # def write_log(self,date):
    #     """"date","type","hold","direction","average_cost"

    #     Args:
    #         date (_type_): _description_
    #     """
    #     for future_type, value in self.position.hold.items():
    #         info = future_type.split("_")
    #         self.hold.loc[len(self.hold)]=[date,info[0],value[0],info[1],value[1]/10000]
    
    def calculate_profit(self,m_data:dict):
        total_profit=0
        for future_type, value in m_data.items():
            try:
                current_close = value["close"].iloc[-1]
            except:
                continue
            preclose = value["prev_close"].iloc[-1]
            
            if (future_type+"_long") in self.position.hold.keys():
                self.position.hold[future_type+"_long"][2]+=(int(current_close*10000)-int(preclose*10000))*self.position.hold[future_type+"_long"][0]
                total_profit+=self.position.hold[future_type+"_long"][2]

            if (future_type+"_short") in self.position.hold.keys():
                self.position.hold[future_type+"_short"][2]-=(int(current_close*10000)-int(preclose*10000))*self.position.hold[future_type+"_short"][0]
                total_profit+=self.position.hold[future_type+"_short"][2]

        
        #通过hold和cash计算收益
        self.position.asset.append(self.position.cash+total_profit)
    
    def statistics(self,profits:pd.DataFrame):
        """_summary_
        [Obsolete]不再这里进行统计了，已废弃
        Args:
            profits (pd.DataFrame): _description_
        """
        profits["delta"]=(profits.shift(-1)[self.context.name]- profits[self.context.name])/profits[self.context.name]
        std_error=np.sqrt(250)*profits["delta"].std()
        self.log(std_error)
        a:pd.Series= profits[self.context.name]
        time=profits["date"]
        v_start=a.iloc[0]
        v_end =a.iloc[-1]
        t_start=time.iloc[0]
        t_end=time.iloc[-1]
        T=(t_end-t_start).days/365.25
        profit=np.log(v_end/v_start)/T
        self.log(f"年收益率{profit*100}%,夏普比率:{profit/std_error}")
        
        
    
    ######以下等待完善，用于因子相关性分析，独立于回测，暂时用不上######    
    def beautiful_plot(self):
        pass
    
    def factor_builder(self,m_data:pd.DataFrame,context):
        pass
    def factor_statistics(self,type):
        m_data=self.data[type].copy()
        m_data:pd.DataFrame=self.factor_builder(m_data,self.context)
        ##相关性测试
        corre = m_data.corr()
        corre.to_excel("correlation.xlsx")
        
        
        
 

        
        
        
    

