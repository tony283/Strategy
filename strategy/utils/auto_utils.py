
import pandas as pd
import random
import torch
from datetime import datetime
class AutoFactorGenerator():
    def __init__(self,depth=10) -> None:
        typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
        self.data={}
        for item in typelist:
            try:
                self.data[item] = pd.read_csv("data/"+item+"_daily.csv",index_col=0)
                self.data[item]["date"]=self.data["date"].apply(lambda x:datetime.strptime(x,"%Y-%m-%d"))
            except Exception as e:
                print(e)
        self.function_pool=FunctionPool(depth)




class FunctionPool:
    def __init__(self,depth) -> None:
        self.calculator=[self.DELAY,self.MA,self.STD,self.DIF,self.SMA,self.PCT,self.SKEW,self.KURT,self.ADD,self.MINUS,self.DIV,self.PROD,self.MIN,self.MAX,self.CORR]
        self.mono=[self.DELAY,self.MA,self.STD,self.DIF,self.SMA,self.PCT,self.SKEW,self.KURT]
        self.bi=[self.ADD,self.MINUS,self.DIV,self.PROD,self.MIN,self.MAX,self.CORR]
        self.value=[['close','high','low','open']]
        self.depth=depth
        pass
    def DELAY(self,df:pd.DataFrame, window):
        return df.shift(window)
    def MA(self,df:pd.DataFrame, window):
        return df.rolling(window=window).mean()
    def STD(self,df:pd.DataFrame, window):
        return df.rolling(window=window).std()
    def DIF(self,df: pd.DataFrame,window):
        return df-df.shift(window)
    def SMA(self,df:pd.DataFrame, window):
        return df.ewm(span=window,adjust=False).mean()
    def PCT(self,df:pd.DataFrame, window):
        return (df-df.shift(window))/df.shift(window)
    def SKEW(self,df:pd.DataFrame, window):
        return df.rolling(window=window).skew()
    def KURT(self,df:pd.DataFrame, window):
        return df.rolling(window=window).kurt()
    def ADD(self,df1,df2):
        return df1+df2
    def MINUS(self,df1,df2):
        return df1-df2
    def DIV(self,df1,df2):
        return df1/df2
    def PROD(self,df1,df2):
        return df1*df2
    def MIN(self,df1,df2):
        return pd.concat([df1,df2],axis=1).min(axis=1)
    def MAX(self,df1,df2):
        return pd.concat([df1,df2],axis=1).max(axis=1)
    ####3####
    def CORR(self,df1,df2,window):
        return df1.rolling(window=window).corr(df2)
    
    def MonoCombine(self):
        random_calculator = self.mono[random.randint(0,len(self.mono)-1)]
        return random_calculator
    def BiCombine(self):
        random_calculator = self.bi[random.randint(0,len(self.bi)-1)]
        return random_calculator
    def RamdomCombine(self):
        random_calculator = self.calculator[random.randint(0,len(self.calculator)-1)]
        return random_calculator

class Node():
    def __init__(self,func=None,name=None,node_type=True,window=None,capacity=0) -> None:
        """_summary_

        Args:
            func (_type_, optional): _description_. Defaults to None.
            value (_type_, optional): _description_. Defaults to None.
            self.type: True-value, False-function

        Raises:
            Warning: _description_
        """
        
        self.node_type=node_type
        self.child_nodes=[]
        if self.node_type:
            self.name=name
            self.func=lambda x:x
            self.capacity=0
        else:
            self.func=func
            self.capacity=capacity
        self.window=window
    def Add(self,node):
        if self.capacity==0:
            return False
        elif self.capacity>len(self.child_nodes):
            self.child_nodes.append(node)
            return True
        else:
            index=0
            l=len(self.child_nodes)
            while index<l:
                if self.child_nodes[index].Add(node):
                    return True
                index+=1
        return False
    def __str__(self):
        if self.capacity==0:
            return self.name
        if self.capacity==1:
            return f'{self.func.__name__}({str(self.child_nodes[0])},{self.window})'
        s=[]
        for i in self.child_nodes:
            s.append(f"({str(i)})")
        return f'{str(self.func.__name__)}[{s[0]} | {s[1]}]'
                
            
    
    def __call__(self,df):
        if self.node_type:
            df[self.__str__()]=df[self.name]
            return df[self.__str__()]
        elif self.capacity==1:
            s=self.child_nodes[0](df)
            a =self.func(s,self.window)
            return a
        elif self.func.__name__=="CORR":
            a = self.func(self.child_nodes[0](df),self.child_nodes[1](df),self.window)
            return a
        else:
            a = self.func(self.child_nodes[0](df),self.child_nodes[1](df))
            return a
def add(x,y):
    return x+y


class XTree():
    def __init__(self,maxsize=10) -> None:
        self.main_node:Node=None
        self.functions=FunctionPool(10)
        self.maxsize=maxsize
        self.names=['close','high','low','open','volume','open_interest']
        self.window=[1,5,20,63,126]
        
    def split(self):
        for i in range(self.maxsize):
            if random.random()<i/self.maxsize:
                signal=self.Add(Node(name=self.names[random.randint(0,5)],capacity=0,window=self.window[random.randint(0,4)]))
            elif random.random()>0.5:
                signal=self.Add(Node(func=self.functions.BiCombine(),node_type=False,capacity=2,window=self.window[random.randint(0,4)]))
            else:
                signal=self.Add(Node(func=self.functions.MonoCombine(),node_type=False,capacity=1,window=self.window[random.randint(0,4)]))
            if not signal:
                break
        while True:
            if not self.Add(Node(name=self.names[random.randint(0,5)],capacity=0)):
                break
        print(str(self.main_node))
    def Add(self,node):
        if self.main_node==None:
            self.main_node=node
            return True
        else:
            return self.main_node.Add(node)
    def __call__(self,df):
        return self.main_node(df)


            
        

            
        




df=pd.read_csv("data/CU_daily.csv")

            
a=XTree(maxsize=10)
a.split()
print(a(df))    
