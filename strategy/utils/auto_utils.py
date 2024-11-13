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
from concurrent.futures import ThreadPoolExecutor
import warnings
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 

class FunctionPool:
    """
    This is a function pool for the use of XTree to randomly select the function needed.
    The function is divided into 2 parts. 
    First is the mono, which uses one dataframe and several other params.
    Second is the bi, which uses two dataframe and several other params.
    """
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
    """
    This is a class of the node of the XTree.
    There are two kind of node in total.
    First is the value node, which requires params: node_type=True, name=notNone and capacity=0
    Second is the operator node, which requires params: node_type=False, func=notNone and capacity not zero.
    Window is a optional param for every operator that requires a window param as input.
    """
    def __init__(self,func=None,name=None,node_type=True,window=None,capacity=0) -> None:
        """_summary_

        Args:
            func (function, optional): operator generated from FunctionPool. Defaults to None.
            name (str, optional): column name of the dataframe. Used by value node. Defaults to None.
            node_type (bool, optional): True refers to value node. False refers to operator node. Defaults to True.
            window (int, optional): optional param for every operator that requires a window param as input. Defaults to None.
            capacity (int, optional): the max length of child_nodes. Defaults to 0.
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
        """_summary_

        Args:
            node (Node): the node ready to be added to the child nodes.

        Returns:
            bool: the node is successfully added to the child_nodes if true else the capacity of the child_nodes is full.
        """
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
        """_summary_

        Returns:
            str: return the name of the node.
        """
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
        """_summary_

        Args:
            df (pd.Dataframe): the raw data as input

        Returns:
            pd.Series: returns the factor calculated by the node.
        """
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
    """
    XTree is a tree structure that allows the random generation of a tree by simply using the XTree.split().
    """
    def __init__(self,maxsize=10) -> None:
        """_summary_

        Args:
            maxsize (int, optional): maxsize restricts the node number of the tree. The number of node  will not be greater than two times of the maxsize. Defaults to 10.
        """
        self.main_node:Node=None
        self.functions=FunctionPool()
        self.maxsize=maxsize
        self.names=['high_close','low_close','sigma_skew20','skew_position63','skew_position20','d_position5','vol_skew20','break20','vol_skew126','corr_ret_vol','vol_kurt126','price_kurt14','position63','relative_amihud5']
        self.name_l=len(self.names)-1
        self.window=[1,5,9,12,20,26,63]
        
    def split(self):
        """
        Randomly generating a tree
        """
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
        logger.info('Generation success: '+str(self.main_node))
    def Add(self,node):
        """_summary_

        Args:
            node (Node): the node ready to add

        Returns:
            bool: return true if the node is successfully added.
        """
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
    """
    the engine for genetic algorithm
    """
    def __init__(self, pop_num,maxsize=5) -> None:
        """

        Args:
            pop_num (int): the number of trees to generate
            maxsize (int, optional): controls the maximum node in a tree. Defaults to 5.
        """
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
        """
        crossover between 2 trees

        Args:
            tree1 (XTree): tree1
            tree2 (XTree): tree2

        Returns:
            XTree, XTree: crossovered tree1 and tree2
        """
        logger.info("Before crossover:")
        logger.info("Tree1:"+str(tree1.main_node))
        logger.info("Tree2:"+str(tree2.main_node))
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

        logger.info("After crossover:")
        logger.info("Tree1:"+str(tree1.main_node))
        logger.info("Tree2:"+str(tree2.main_node))

        return tree1, tree2
    def calculate_fitness(self):
        """
        calculate the fitness of each population.

        Returns:
            np.ndarray: the fitness array
        """
        fitness = []
        l=len(self.population)
        #随机抽取训练集
        random_data=random.sample(self.data,int(0.1*len(self.data)))
        for  i in range(l):
            corr=pd.DataFrame()
            logger.debug(str(self.population[i]))
            pop=self.population[i]
            for df in random_data:
                df['a']=pop(df)
                df=df[['a','expect1']]
                
                if len(corr)==0:
                    corr=df.corr()
                else:
                    corr=corr+df.corr()
                # x=df[['a','expect1']].to_numpy()
                # x=np.nan_to_num(x)
                # temp = np.ma.corrcoef(x,rowvar=0)[0, 1]
                # if temp != temp:
                #     continue
                # corr+=temp
            corr=np.abs(corr.loc['expect1','a']/len(self.data))
            # adaption=np.abs(corr.loc['expect1','a'])
            adaption=corr
            
            fitness.append(adaption)
            if (i%200==0):
                print(f'\r{(100*i/l):.2f}% fit',end='\r')
        return np.nan_to_num(np.array(fitness))
    
    def calculate_all_fitness(self):
        """
        calculate the fitness of each population.

        Returns:
            np.ndarray: the fitness array
        """
        fitness = []
        l=len(self.population)

        for  i in range(l):
            corr=pd.DataFrame()
            logger.debug(str(self.population[i]))
            pop=self.population[i]
            for df in self.data:
                df['a']=pop(df)
                df=df[['a','expect1']]
                
                if len(corr)==0:
                    corr=df.corr()
                else:
                    corr=corr+df.corr()
                # x=df[['a','expect1']].to_numpy()
                # x=np.nan_to_num(x)
                # temp = np.ma.corrcoef(x,rowvar=0)[0, 1]
                # if temp != temp:
                #     continue
                # corr+=temp
            corr=np.abs(corr.loc['expect1','a']/len(self.data))
            # adaption=np.abs(corr.loc['expect1','a'])
            adaption=corr
            
            fitness.append(adaption)
            if (i%200==0):
                print(f'\r{(100*i/l):.2f}% fit',end='\r')
        return np.nan_to_num(np.array(fitness))
    

    
    
    def mutate(self, tree: XTree):
        """随机突变 tree 中的一个节点"""
        path = [random.randint(0, len(tree.main_node.child_nodes) - 1)]
        node = tree.find_node(path)
        logger.info(f'before mutation:{tree}')
        while node and node.capacity > 0:
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
        
        logger.info("After mutation:"+str(tree))
        return tree

    def crossover_population_with_selection(self, population, fitness):
        """
        give each population a prob according to the fitness and randomly generates children throuth the selection based on the prob.

        Args:
            population (list(XTree)): populations
            fitness (np.ndarray): fitness array

        Returns:
            list(XTree): new_population
        """
        total_fitness = sum(fitness)
        selection_probs = [f / total_fitness for f in fitness]

        new_population = []
        new_population.append(self.population[np.argmax(total_fitness)])
        while len(new_population) < len(population):
            parent1 = np.random.choice(population, p=selection_probs)
            parent2 = np.random.choice(population, p=selection_probs)

            while parent1 == parent2:
                parent2 = np.random.choice(population, p=selection_probs)

            child1, child2 = self.crossover(parent1, parent2)
            new_population.extend([child1, child2])

        return new_population[:len(population)]


    
    def loop(self):
        """
        one loop includes calculating fitness, crossover and mutation.
        """
        fitness=self.calculate_fitness()
        logger.info(f'fitness is {fitness}')
        #交叉
        new_population = self.crossover_population_with_selection(self.population,fitness)
        self.population=new_population
        #开始变异
        idx=list(range(len(self.population)))
        indexes=random.sample(idx,int(0.1*len(self.population)))
        for i in indexes:
            self.population[i]=self.mutate(self.population[i])
        return np.max(fitness)
    def run(self,generation=10):
        # warnings.simplefilter("ignore", category=RuntimeWarning)
        """
        loop circulation. The main interface of genetic_algorithm.

        Args:
            generation (int, optional): the num of loops. Defaults to 10.
        """
        for i in range(generation):
            fit=self.loop()
            print(f"Best fitness: {fit}")
            if fit>0.05:
                break
            print(f' generation {i} of {generation} Completed')
        fitness = self.calculate_all_fitness()
        df=[[x,_] for _, x in sorted(zip(fitness, self.population), reverse=True,key=lambda x: x[0])]
        df=pd.DataFrame(df,columns=['factor','fitness'])
        df=df.drop_duplicates(subset=['factor'])
        df.to_excel(f'factor/auto/auto_factor_pop{len(self.population)}_depth{self.maxsize}.xlsx')
        
        
if __name__=='__main__':
    g=genetic_algorithm(6000,maxsize=6)
    t=time.time()
    g.run(10)
    print(time.time()-t)


# df=pd.DataFrame()
# df.to_excel('factor/auto/test,.xlsx')
# a=np.zeros(10)
# b=np.nan_to_num(np.corrcoef(a,a))
# print(b)