'''
:@Author: LRF
:@Date: 11/12/2024, 10:54:07 AM
:@LastEditors: LRF
:@LastEditTime: 11/12/2024, 10:54:07 AM
:Description: 
'''

import time
import pandas as pd
import random
import torch
import numpy as np
from datetime import datetime
import copy
from multiprocessing import Pool
import logging
import numba
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 

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
                logger.error(e)
        self.function_pool=FunctionPool(depth)




class FunctionPool:
    def __init__(self) -> None:
        self.calculator=[self.RANK,self.DELAY,self.MA,self.STD,self.DIF,self.SMA,self.PCT,self.SKEW,self.KURT,self.RMIN,self.RMAX,self.ADD,self.MINUS,self.DIV,self.PROD,self.MIN,self.MAX,self.CORR]
        self.mono=[self.RANK,self.DELAY,self.MA,self.STD,self.DIF,self.SMA,self.PCT,self.SKEW,self.KURT,self.RMIN,self.RMAX]
        self.bi=[self.ADD,self.MINUS,self.DIV,self.PROD,self.MIN,self.MAX,self.CORR]

    def DELAY(self,df:pd.DataFrame, window):
        return df.shift(window)
    def MA(self,df:pd.DataFrame, window):
        return df.rolling(window=window).mean()
    def RANK(self,df:pd.DataFrame, window):
        return df.rolling(window=window).rank()
    def STD(self,df:pd.DataFrame, window):
        return df.rolling(window=window).std()
    def DIF(self,df: pd.DataFrame,window):
        return df-df.shift(window)
    def SMA(self,df:pd.DataFrame, window):
        return df.ewm(span=window,adjust=False).mean()
    def PCT(self,df:pd.DataFrame, window):
        return (df-df.shift(window))/df.shift(window)
    def RMIN(self,df:pd.DataFrame, window):
        return df.rolling(window=window).min()
    def RMAX(self,df:pd.DataFrame, window):
        return df.rolling(window=window).max()
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
            return f'{self.func.__name__}{self.window}({str(self.child_nodes[0])})'
        s=[]
        for i in self.child_nodes:
            s.append(f"{str(i)}")
        if self.func.__name__=="CORR":
            return f'CORR{self.window}[{s[0]} , {s[1]}]'
        return f'{str(self.func.__name__)}[{s[0]} , {s[1]}]'
                
            

    def __call__(self,df):
        if self.node_type:
            # df[self.__str__()]=df[self.name]
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


class XTree():
    def __init__(self,maxsize=10) -> None:
        self.main_node:Node=None
        self.functions=FunctionPool()
        self.maxsize=maxsize
        self.names=['high_close','low_close','sigma_skew20','skew_position63','skew_position20','d_position5','vol_skew20','break20','vol_skew126','corr_ret_vol','vol_kurt126','price_kurt14','position63','relative_amihud5']
        self.name_l=len(self.names)-1
        self.window=[1,5,9,12,20,26,63]
        
    def split(self):
        for i in range(self.maxsize):
            if random.random()<i/self.maxsize:
                signal=self.Add(Node(name=self.names[random.randint(0,self.name_l)],capacity=0,window=self.window[random.randint(0,6)]))
            elif random.random()>0.4:
                signal=self.Add(Node(func=self.functions.BiCombine(),node_type=False,capacity=2,window=self.window[random.randint(1,6)]))
            else:
                function = self.functions.MonoCombine()
                if function.__name__ in ['KURT','SKEW',"STD",'RMIN','RMAX','MA','SMA',"RANK"]:
                    window_index=random.randint(1,6)
                else:
                    window_index=random.randint(0,6)
                signal=self.Add(Node(func=function,node_type=False,capacity=1,window=self.window[window_index]))
            if not signal:
                break
        while True:
            if not self.Add(Node(name=self.names[random.randint(0,self.name_l)],capacity=0)):
                break
        logger.debug('Generation success: '+str(self.main_node))
    def Add(self,node):
        if self.main_node==None:
            self.main_node=node
            return True
        else:
            return self.main_node.Add(node)
    def find_node(self, path):
        """Finds the node at the given path in the tree."""
        current_node = self.main_node
        for index in path:
            if index < len(current_node.child_nodes):
                current_node = current_node.child_nodes[index]
            else:
                return None  # 路径错误，返回 None
        return current_node

    def replace_node(self, path, new_node):
        """Replaces the node at the given path with new_node."""
        if not path:
            self.main_node = new_node  # 如果路径为空，则替换根节点
        else:
            parent_path, index = path[:-1], path[-1]
            parent_node = self.find_node(parent_path)
            if parent_node and index < len(parent_node.child_nodes):
                parent_node.child_nodes[index] = new_node
    def __call__(self,df):
        return self.main_node(df)
    def __str__(self) -> str:
        return str(self.main_node)




class genetic_algorithm:
    def __init__(self, pop_num,maxsize=5) -> None:
        self.typelist= ['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
        self.data=[]
        self.population=[]
        self.maxsize=maxsize
        self.names=['high_close','low_close','sigma_skew20','skew_position63','skew_position20','d_position5','vol_skew20','break20','vol_skew126','corr_ret_vol','vol_kurt126','price_kurt14','position63','relative_amihud5']
        for i in range(pop_num):
            a=XTree(maxsize=maxsize)
            a.split()
            self.population.append(a)
        for i in self.typelist:
            future_data=pd.read_csv(f'data/{i}_daily.csv')
            future_data["date"]=future_data["date"].apply(lambda x:datetime.strptime(x,"%Y-%m-%d"))
            self.data.append(future_data[future_data['date']>datetime(2018,1,1)])
    def crossover(self,tree1:XTree,tree2:XTree):
        logger.debug("Before crossover:")
        logger.debug("Tree1:"+str(tree1.main_node))
        logger.debug("Tree2:"+str(tree2.main_node))
        path1 = [random.randint(0, len(tree1.main_node.child_nodes) - 1)]
        path2 = [random.randint(0, len(tree2.main_node.child_nodes) - 1)]
        node1 = tree1.find_node(path1)
        while node1 and node1.capacity > 0 and random.random() > 0.6:
            idx = random.randint(0, len(node1.child_nodes) - 1)
            path1.append(idx)
            node1 = node1.child_nodes[idx]

        node2 = tree2.find_node(path2)
        while node2 and node2.capacity > 0 and random.random() > 0.6:
            idx = random.randint(0, len(node2.child_nodes) - 1)
            path2.append(idx)
            node2 = node2.child_nodes[idx]

        c_node1 = copy.deepcopy(node1)
        c_node2 = copy.deepcopy(node2)

        tree1.replace_node(path1, c_node2)
        tree2.replace_node(path2, c_node1)

        logger.debug("After crossover:")
        logger.debug("Tree1:"+str(tree1.main_node))
        logger.debug("Tree2:"+str(tree2.main_node))

        return tree1, tree2
    def calculate_fitness(self):
        
        fitness = []
        l=len(self.population)
        for  i in range(l):
            corr=pd.DataFrame()
            logger.info(str(self.population[i]))
            pop=self.population[i]
            for df in self.data:
                df['a']=pop(df)
                df=df[['a','expect1']]
                if len(corr)==0:
                    corr=df.corr()
                else:
                    corr=corr+df.corr()
            corr=corr/len(self.typelist)
            adaption=np.abs(corr.loc['expect1','a'])
            
            fitness.append(adaption)
            if (i%200==0):
                print(f'\r{100*i/l}% fit')
        return np.nan_to_num(np.array(fitness))
    def mutate(self, tree: XTree):
        """随机突变 tree 中的一个节点"""
        path = [random.randint(0, len(tree.main_node.child_nodes) - 1)]
        node = tree.find_node(path)
        logger.debug(f'before mutation:{tree}')
        while node and node.capacity > 0 and random.random() > 0.1:
            idx = random.randint(0, len(node.child_nodes) - 1)
            path.append(idx)
            node = node.child_nodes[idx]
        
        # 进行突变：随机改变节点类型、窗口参数或替换为新函数
        if node.node_type:
            # 如果是值节点，可以改变它的字段名称
            node.name = random.choice(self.names)
        else:
            # 如果是函数节点，随机改变窗口或替换函数
            node.func = random.choice(tree.functions.mono if node.capacity == 1 else tree.functions.bi)
            if node.func.__name__ in ['KURT','SKEW',"STD",'RMIN','RMAX','CORR','SMA']:
                node.window=tree.window[random.randint(1,len(tree.window)-1)]
            else:
                node.window = random.choice(tree.window)
        
        logger.debug("After mutation:"+str(tree))
        return tree

    def crossover_population_with_selection(self, population, fitness):
        # 根据适应度分配选择概率
        total_fitness = sum(fitness)
        selection_probs = [f / total_fitness for f in fitness]

        new_population = []
        while len(new_population) < len(population):
            # 根据选择概率随机选择两个个体作为父母
            parent1 = np.random.choice(population, p=selection_probs)
            parent2 = np.random.choice(population, p=selection_probs)

            # 确保两个父母不相同
            while parent1 == parent2:
                parent2 = np.random.choice(population, p=selection_probs)

            # 进行交叉生成子代
            child1, child2 = self.crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # 裁剪新种群以适应原种群大小
        return new_population[:len(population)]


    def calculate_single_fitness(self,pop):
        
        corr=pd.DataFrame()
        logger.debug(str(pop))
        for df in self.data.values():
            df['a']=pop(df)
            df=df[['a','expect1']]
            if len(corr)==0:
                corr=df.corr()
            else:
                corr=corr+df.corr()
        corr=corr/len(self.typelist)
        adaption=np.abs(corr.loc['expect1','a'])
        if adaption!=adaption:
            return 0
        return adaption


    
    def loop(self):
        fitness=self.calculate_fitness()
        # fitness=self.parallel_fitness(self.population)
        logger.debug(f'fitness is {fitness}')
        #交叉
        new_population = self.crossover_population_with_selection(self.population,fitness)
        self.population=new_population
        #开始变异
        idx=list(range(len(self.population)))
        indexes=random.sample(idx,int(0.2*len(self.population)))
        for i in indexes:
            self.population[i]=self.mutate(self.population[i])
    def run(self,generation=10):
        for i in range(generation):
            self.loop()
            print(f'\r {100*i/generation}% Completed')
        fitness = self.calculate_fitness()
        df=[[x,_] for _, x in sorted(zip(fitness, self.population), reverse=True,key=lambda x: x[0])]
        df=pd.DataFrame(df,columns=['factor','fitness'])
        df=df.drop_duplicates(subset=['factor'])
        df.to_excel(f'factor/auto/auto_factor_pop{len(self.population)}_depth{self.maxsize}.xlsx')
        
        
if __name__=='__main__':
    g=genetic_algorithm(5000,maxsize=7)
    t=time.time()
    g.run(20)
    print(time.time()-t)


# df=pd.DataFrame()
# df.to_excel('factor/auto/test,.xlsx')