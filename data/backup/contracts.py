from typing import Any
import rqdatac
import pandas as pd
rqdatac.init()
from datetime import datetime
from rescale import *
import os
import json
import sys
def timer(func):
    """_summary_
    装饰器计时
    Args:
        func (_type_): _description_
    """
    def func_wrapper(*args,**kwargs):
        from time import time
        time_start = time()
        result = func(*args,**kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper
def str2list(s:str):
    """_summary_
        将str反序列化成list
    Args:
        s (str): _description_

    Returns:
        _type_: _description_
    """
    s=s.replace("[","")
    s=s.replace("]","")
    s=s.replace("'","")
    l = s.split(",")
    l =[i.strip() for i in l]
    return l
    

def generate_list(dominant:str,contracts:list,switchback:bool)->str:
    """_summary_
    根据是否回切生成合约列表
    Args:
        dominant (str): 当前主力合约
        contracts (list): 当前日期合约列表
        switchback (bool): 是否回切

    Returns:
        _type_: _description_
    """
    contract_list =[]
    if switchback:
        contract_list =  contracts
    else:
        flag = True
            
        for i in contracts:
            
            if flag and i==dominant:
                flag=False
            if not flag:
                contract_list.append(i)
    return contract_list

def read_contracts(future_type,start,end):
    """_summary_
    生成每日所有未到期contract的dataframe，从米筐中获取数据

    Args:
        future_type (_type_): 期货代码
        start (_type_): 起始日
        end (_type_): 终止日
    """
    date = rqdatac.futures.get_dominant(future_type,start_date=start,end_date=end)
    contracts = pd.DataFrame(index=date.index)
    for i in date.index:
        contracts.loc[i,"contracts"] = rqdatac.futures.get_contracts(future_type, date=i)
    contracts.to_excel(f"data/{future_type}/{future_type}_contracts.xlsx")



class Contracts:
    def __init__(self) -> None:
        self.data = {}
            
    def process_contracts(self,future_type,start,end,switchback,*conditions):
        start,end = datetime.strptime(start,"%Y%m%d"),datetime.strptime(end,"%Y%m%d")
        contracts = pd.read_excel(f"data/{future_type}/{future_type}_contracts.xlsx",index_col=0)
        contracts = contracts.loc[start:end]#将日期切片到所需范围
        contracts["contracts"]=contracts["contracts"].apply(lambda x:str2list(x))#将str转化为list
        current_dominant = rqdatac.futures.get_dominant(future_type,contracts.index.to_list()[0])[0]#获取起始日dominant
        contracts["dominant_list"]=None
        contracts["swith"] = 0
        contracts["dominant"] = None
        length = len(contracts)
        count=0
        for i in contracts.index:
            count+=1
            domiant_list = self.dominant(i,contracts.loc[i,"contracts"],current_dominant,conditions,switchback)#获得当日主力次主力合约列表
            contracts.loc[i,"dominant_list"]=str(domiant_list)
            if domiant_list[0]!=current_dominant:
                contracts.loc[i,"switch"]=1
                current_dominant=domiant_list[0]#主力发生切换
            else:
                contracts.loc[i,"switch"]=0
            contracts.loc[i,"dominant"]=current_dominant
            if(count%10==0):#显示输出
                print("\r", end="")
                print(f"Progress{count*100/length:.2f}%",end="")
                sys.stdout.flush()
        
        contracts.to_excel(f"data/{future_type}/{future_type}_dominant_result.xlsx")
                    
    ####主函数####

    def dominant(self,date, contracts: list,current_dominant,conditions,switchback:bool=False):
        """_summary_

        Args:
            date (datetime): 日期
            contracts (list): 当前日所有合约列表
            current_dominant (str): 上一日的主力合约
            conditions (_type_): 一系列判断条件
            switchback (bool, optional): 是否允许回切. Defaults to False.

        Raises:
            SyntaxError: _description_

        Returns:
            _type_: _description_
        """
        dominant_list = []
        if current_dominant not in contracts:
            print(f"Cannot find {current_dominant} in {contracts}")
            raise SyntaxError
        checklist = generate_list(current_dominant,contracts,switchback)#根据是否回切形成checklist
        for i in checklist:
            if self.check_condition(date, i,*conditions):#对checklist中每个合约进行条件检测，将满足条件的合约加入到dominant_list
                dominant_list.append(i)
        return dominant_list

    def try_get_contract_daily(self,contract_name:str):
        """_summary_

        Args:
            contract_name (str): 合约代码

        Returns:
            _type_: 合约历史数据
        """
        month = int(contract_name[-2:])
        year = int(contract_name[-4:-2])
        if not os.path.exists(f"data/{contract_name[:-4]}/"):
            os.makedirs(f"data/{contract_name[:-4]}/")#没有创建过文件夹则创建
        if not os.path.exists(f"data/{contract_name[:-4]}/{contract_name}.xlsx"):
            rqdatac.get_price(contract_name,adjust_type="post").to_excel(f"data/{contract_name[:-4]}/{contract_name}.xlsx")#再从米筐读数据存入本地
        elif year<datetime.today().year or (year==datetime.today().year and month<datetime.today().month):
            pass
        else:
            rqdatac.get_price(contract_name,adjust_type="post").to_excel(f"data/{contract_name[:-4]}/{contract_name}.xlsx")#如果合约还未到期，需要实时更新
        try:
            return self.data[contract_name]#尝试从运行中的内存获取
        except:
            self.data[contract_name] = pd.read_excel(f"data/{contract_name[:-4]}/{contract_name}.xlsx")#否则重新从本地读取
            return self.data[contract_name]
  
    def try_get_contract_instrument(self,contract_name:str):
        """_summary_
            获取期货合约具体信息
        Args:
            contract_name (str): 合约代码

        Returns:
            _type_: 合约具体信息(字典格式)
        """
        info = None

        if not os.path.exists(f"data/{contract_name[:-4]}/"):
            os.makedirs(f"data/{contract_name[:-4]}/")#如果没获取过创建文件夹
        if not os.path.exists(f"data/{contract_name[:-4]}/{contract_name}.txt"):
            with open(f'data/{contract_name[:-4]}/{contract_name}.txt', 'w') as f:
                f.write(json.dumps(rqdatac.instruments(contract_name).__dict__))#如果获取过将其转为json
        try:
            info = self.data[f"{contract_name}_instrument"]#尝试从已有数据获取
        except:
            with open(f'data/{contract_name[:-4]}/{contract_name}.txt', 'r') as f:
                info = json.loads(f.read())
                self.data[f"{contract_name}_instrument"]=info
        return info
    def check_condition(self,date,contract_name,*args)-> bool:
        """_summary_
        检查是否满足条件

        Args:
            date (_type_): _description_
            contract_name (_type_): 要检查的合约名称
            args：可以在此增加条件

        Returns:
            bool: _description_
        """
        for func in args:
            if func(date, contract_name,self):
                continue
            else:
                return False
        return True
    



def working_days(start,end):
    """_summary_

    Args:
        start (_type_): 开始日期
        end (_type_): 结束日期

    Returns:
        _type_: 返回两个日期间的工作日天数
    """
    try:
        start = datetime.fromtimestamp(start)#格式化
    except:
        pass
    try:
        end = datetime.fromtimestamp(end)#格式化
    except:
        pass
        
    dates = pd.read_excel("data/trading_dates.xlsx",index_col=0)
    dates = dates[dates["date"]<end]
    dates = dates[dates["date"]>=start]
    return len(dates)



class Condition:
    """_summary_自定义类，需要有call函数，返回值是bool值
    """
    def __init__(self):
        pass
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return True


 #############################################################################################   
class VolumeCondition:
    """_summary_成交量条件
    """
    def __init__(self,max_volume,days:int):
        self.max_volume = max_volume
        self.days = days
        pass
    def __call__(self, date, contract_name, m_contract:Contracts, *args: Any, **kwds: Any) -> bool:
        data = m_contract.try_get_contract_daily(contract_name)
        if len(data[data["date"]<date])<self.days:
            return False
        return data[data["date"]<date][-self.days:]["volume"].min()>=self.max_volume

class TurnoverCondition:
    """_summary_成交额条件
    """
    def __init__(self,max_turnover,days:int):
        self.max_turnover = max_turnover
        self.days = days
        pass
    def __call__(self, date, contract_name, m_contract:Contracts,*args: Any, **kwds: Any) -> bool:
        data = m_contract.try_get_contract_daily(contract_name)
        if len(data[data["date"]<date])<self.days:
            return False
        return data[data["date"]<date][-self.days:]["total_turnover"].min()>=self.max_turnover
    
class Open_IntrestCondition:
    """_summary_持仓量条件
    """
    def __init__(self,max_interest,days:int):
        self.max_turnover = max_interest
        self.days = days
        pass
    def __call__(self, date, contract_name,m_contract:Contracts, *args: Any, **kwds: Any) -> bool:
        data = m_contract.try_get_contract_daily(contract_name)
        if len(data[data["date"]<date])<self.days:
            return False
        return data[data["date"]<date][-self.days:]["open_interest"].min()>=self.max_turnover
######################################################################################################
class DeListedCondition:
    """_summary_到期日条件
    """
    def __init__(self,days:int,working_day=True):
        self.days = days
        self.working_day = working_day
        
    def __call__(self, date:datetime, contract_name, m_contract:Contracts,*args: Any, **kwds: Any) -> bool:
        end_date = datetime.strptime(m_contract.try_get_contract_instrument(contract_name)["de_listed_date"])
        delta_date = working_days(date,end_date,"%Y-%m-%d") if self.working_day else  end_date-date
        return (delta_date >=self.days)
    
class DeliveryMonthCondition:
    """_summary_交割月条件
    """
    def __init__(self,days:int,working_day=True):
        self.days = days
        self.working_day = working_day 
    def __call__(self, date:datetime, contract_name, m_contract:Contracts,*args: Any, **kwds: Any) -> bool:
        year = int(contract_name[-4:-2])+2000
        month = int(contract_name[-2:])
        end_date = datetime(year,month,1)
        delta_date = working_days(date,end_date) if self.working_day else  end_date-date
        return (delta_date >=self.days)


def handle_contract(future_type,start,end,*conditions):
    """
    主函数，传入期货代码和时间
    """    
    read_contracts(future_type,start,end)    
    test = Contracts() 
    test.process_contracts(future_type,start,end,False,
                    *conditions##可以在后面继续增加条件，也可以自定义条件
                    )

    m_data =pd.read_excel(f"data/{future_type}/{future_type}_dominant_result.xlsx")
    # print(m_data.iloc[0])
    transfer_dominant(m_data,future_type)

if __name__=="__main__":
    handle_contract("CU","20120102","20240827")
