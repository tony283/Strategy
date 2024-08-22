
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import threading
def try_set_value(a:dict,key,value):
    """_summary_

    Args:
        a (dict): _description_
        key (_type_): _description_
        value (_type_): [amount,price,margin]
    """
    assert value[0]>=0
    if (key in a.keys()):
        originial_hold =a[key][0]
        a[key][0]+=value[0]
        
        a[key][1]=(value[0]*value[1]+originial_hold*a[key][1])//a[key][0]
        a[key][2]+=value[2]
    else:
        a[key]=np.array(value)
def try_sell_value(a:dict,key,value,direction):
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
    
    if real_key in a.keys():
        earn=  int(a[real_key][2])*int(value[0])//int(a[real_key][0])
        a[real_key][2] -=earn#amount*close_price
        a[real_key][0] -=value[0]
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
        self.context=context()
        self.debug = debug
        self.data = {}
        self.position = context()
        self.position.cash=cash*10000#一切财务以0.01分作为最小单元
        self.position.original_cash = cash*10000#参考基准
        self.position.hold = {}#hold item格式：[amount,price*10000,guarantee]
        self.position.asset=[]
        
        #self.hold =pd.DataFrame(columns=["date","type","hold","direction","average_cost"])#历史持仓，太耗时，已取消
        self.trade_record = pd.DataFrame(columns=["date","type","amount","direction","B/S","price"])#成交记录
        self.instrument = {"margin_rate":margin_rate,"margin_limit":margin_limit}
        self.init(self.context)
        pass
    def init(self,context):
        pass
        pass
    def subscribe(self,future_type:str):
        try:
            future_data = pd.read_excel("data/"+future_type+"_daily.xlsx")
            future_data
        except:
            raise ReadingError("Cannot load data/"+future_type+"_daily.xlsx")
        self.data[future_type]=future_data
        
    def log(self,s:str):
        if (not self.debug):
            pass
        else:
            print(s)
        
    def before_trade(self, context):
        pass
    def after_trade(self,context):
        pass
    def handle_bar(self,m_data:pd.DataFrame,context):
        """_summary_

        Args:
            m_data (Dataframe): 这是到当前日为止的所有数据，注意回测的时候只能用最后一行的收盘价作为买入价，其余数据计算应当从倒数第二行开始，防止未来函数
            context (_type_): _description_
        """
        pass
    
    def process(self,m_data):
        self.before_trade(self.context)
        self.check_hold(m_data)#补交保证金
        self.handle_bar(m_data,self.context)
        self.after_trade(self.context)
    def check_hold(self,m_data):
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
        assert direction=="long" or direction=="short"
        margin =int(price*10000*multiplier*amount*self.instrument["margin_rate"])
        if(margin>self.position.cash):
            warnings.warn("保证金不足，无法下单")
            return
        self.log(f"Bid-close:{price},order:{amount*multiplier}")
        try_set_value(self.position.hold,future_type+"_"+direction,np.array([amount*multiplier,
                                                                                 int(price*10000),
                                                                                 margin]
                                                                                )
                          )
        self.position.cash-=margin
        self.trade_record.loc[len(self.trade_record)]=[self.current,future_type,amount*multiplier,direction,"B",price]
        return
    def sell_target_num(self,price,amount:int,multiplier:int,future_type:str,direction):
        assert direction=="long" or direction=="short"
        if(amount<=0):
            warnings.warn(f"{self.current}:卖单量为0，无法完成交易，交易品种{future_type},交易数量{amount}")
            return
        if(True):
            self.log(f"ofr-close:{price},amount:{amount*multiplier}")
            earn=try_sell_value(self.position.hold,future_type,
                           np.array([amount*multiplier,int(price*10000)]),
                           direction
                          )
            self.position.cash+=earn
            self.trade_record.loc[len(self.trade_record)]=[self.current,future_type,amount*multiplier,direction,"S",price]
        return
    #回测
    @timer
    def loop_process(self,start,end):
        time_series:pd.DataFrame = self.data[list(self.data.keys())[0]]["date"]
        time_series=time_series[time_series<=datetime.strptime(end,"%Y%m%d")]
        start_date = datetime.strptime(start,"%Y%m%d")
        real_time_series =time_series[time_series>=start_date]
        current_date=time_series[time_series>=start_date].iloc[0]
        m_data={}#m_data拥有到今日的最新信息，但是策略读取的数据必须从-2往回
        for future_type, value in self.data.items():
            m_data[future_type] = value[value["date"]<current_date]#初始化start日期前的数据
        for i in range(len(real_time_series)):
            current_date=real_time_series.iloc[i]
            self.current=current_date
            for future_type, value in self.data.items():
                m_data[future_type]= value[value["date"]<=current_date]#每天先把当天数据导入，将close作为可以买卖的价格
            self.calculate_profit(m_data)#计算当日收益（分别计算每个品种看涨看跌的收益，将当日价格减去昨日价格）
            self.process(m_data)#进行买卖操作
            #self.write_log(current_date)
        self.log(self.trade_record)
        real_time_series:pd.DataFrame=real_time_series.to_frame()
        real_time_series[self.context.name]=np.array(self.position.asset)
        real_time_series[self.context.name]=real_time_series[self.context.name].apply(lambda x:x/10000)
        real_time_series.to_excel("back/Back_"+self.context.name+".xlsx")
        #self.draw(self.context,real_time_series)
        #self.draw(self.context,real_time_series)
        #self.beautiful_plot()
        #self.statistics(real_time_series)
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
        
        
    
    ######等待完善######    
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
        
        
        
 

        
        
        
    

