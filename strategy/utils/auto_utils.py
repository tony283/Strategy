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
    The special operator CORR uses two dataframe and one window, thus it is specially processed.
    To add a new operator, an operator should be defined first just like the other operators. After that, it should be added to the self.calculator.
    Finally, if it operates on one dataframe, then it should be added to self.mono, otherwise it should be addedd to self.bi.
    One more thing to notice is that some mono operators do not support the param {window=1}, pd.rolling(window=1).std() for example. 
    Although it is okay to get an all-nan array with the bad-parameter operator, it saves time of searching wrong operator.
    Thus, you should exclude it in the the XTree.exclusion.

    """
    
    def __init__(self) -> None:
        self.calculator=[self.RANK,self.DELAY,self.MA,self.STD,self.DIF,self.SMA,self.PCT,self.SKEW,self.KURT,self.RMIN,self.RMAX,self.ADD,self.MINUS,self.DIV,self.PROD,self.MIN,self.MAX,self.CORR]
        self.mono=[self.RANK,self.DELAY,self.MA,self.STD,self.DIF,self.SMA,self.PCT,self.SKEW,self.KURT,self.RMIN,self.RMAX]
        self.bi=[self.ADD,self.MINUS,self.DIV,self.PROD,self.MIN,self.MAX,self.CORR]
    #######The following are the operators#######
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
    ####Special function####
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
        #返回值代表是否添加成功
        if self.capacity==0:
            #如果该节点已经是值节点，无法继续添加子节点
            return False
        elif self.capacity>len(self.child_nodes):
            #如果子节点数小于capacity，则有空位，直接添加
            self.child_nodes.append(node)
            return True
        else:
            #如果子节点满了，在子节点上执行递归查找
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
            #如果是值节点，说明到达底部，直接返回
            return self.name
        if self.capacity==1:
            #如果是mono operator,返回Func{window}(子节点名称)
            return f'{self.func.__name__}{self.window}({str(self.child_nodes[0])})'
        s=[]
        
        for i in self.child_nodes:
            s.append(f"{str(i)}")
        #CORR 特殊处理，返回CORR{window}[子节点1名称,子节点2名称]
        if self.func.__name__=="CORR":
            return f'CORR{self.window}[{s[0]} , {s[1]}]'
        #正常bi operator，返回 Func[子节点1名称,子节点2名称]
        return f'{str(self.func.__name__)}[{s[0]} , {s[1]}]'
                
            

    def __call__(self,df):
        """_summary_

        Args:
            df (pd.Dataframe): the raw data as input

        Returns:
            pd.Series: returns the factor calculated by the node.
        """
        if self.node_type:
            # 如果是值节点，直接返回输入dataframe中列名为self.name的列
            return df[self.__str__()]
        elif self.capacity==1:
            #如果是mono operator，返回 func(子节点所算出的Series，window)
            s=self.child_nodes[0](df)
            a =self.func(s,self.window)
            return a
        elif self.func.__name__=="CORR":
            #如果是CORR，返回 func(子节点1所算出的Series，子节点2所算出的Series，window)
            a = self.func(self.child_nodes[0](df),self.child_nodes[1](df),self.window)
            return a
        else:
            #如果是bi operator，返回 func(子节点1所算出的Series，子节点2所算出的Series)
            a = self.func(self.child_nodes[0](df),self.child_nodes[1](df))
            return a


class XTree():
    """
    XTree is a tree structure that allows the random generation of a tree by simply using the XTree.split().
    It is an encapsulation of node which allows some functions like find, replace, random initialization.
    """
    def __init__(self,maxsize=10) -> None:
        """_summary_

        Args:
            maxsize (int, optional): maxsize restricts the node number of the tree. The number of node  will not be greater than two times of the maxsize. Defaults to 10.
        """
        #根节点
        self.main_node:Node=None
        self.functions=FunctionPool()
        self.maxsize=maxsize
        #sel.names表明所有可以作为值节点的列，可以随意增加减少，只要dataframe中存在同名的列
        self.names=['high_close','low_close','sigma_skew20','skew_position63','skew_position20','d_position5','vol_skew20','break20','vol_skew126','corr_ret_vol','vol_kurt126','price_kurt14','position63','relative_amihud5']
        self.name_l=len(self.names)-1
        #window，可以随意添加，但保证第一个元素是1
        self.window=[1,5,9,12,20,26,63]
        self.window_l=len(self.window)-1
        #如果mono operator不支持window=1，那么需要将其填入self.exclusion
        self.exclusion=['KURT','SKEW',"STD",'RMIN','RMAX','MA','SMA',"RANK"]
    def split(self):
        """
        Randomly generating the main_node
        """
        for i in range(self.maxsize):
            
            if random.random()<i/self.maxsize: # 退火算法，由于值节点会降低生成树的熵，所以最开始值节点不会生成，随着节点数增加值节点形成概率越大
                signal=self.Add(Node(name=self.names[random.randint(0,self.name_l)],capacity=0,window=self.window[random.randint(0,self.window_l)]))
            elif random.random()>0.4: # 概率控制是否生成bi operator还是mono operator
                # 生成 bi
                signal=self.Add(Node(func=self.functions.BiCombine(),node_type=False,capacity=2,window=self.window[random.randint(1,self.window_l)]))
            else: 
                # 生成mono
                function = self.functions.MonoCombine()
                if function.__name__ in self.exclusion:
                    window_index=random.randint(1,self.window_l)
                else:
                    window_index=random.randint(0,self.window_l)
                signal=self.Add(Node(func=function,node_type=False,capacity=1,window=self.window[window_index]))
                
            # signal表示main_node以及其各个子节点中是否还有capacity可以插入，如果没有，说明树已经满了直接结束
            if not signal:
                break
        #有概率树选择了很多bi导致没有完全满，因此，需要用值节点封住所有没满的节点
        while True:
            if not self.Add(Node(name=self.names[random.randint(0,self.name_l)],capacity=0)):
                break
        logger.info('Generation success: '+str(self.main_node))
    def Add(self,node):
        """_summary_
        It is only the encapsulation of Node.Add()

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
        """
        Finds the node at the given path in the tree.
        The path is a list of indexes which points to the index of child_node
        """
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
            parent_node = self.find_node(parent_path) # 先找到parent，否则无法修改
            if parent_node and index < len(parent_node.child_nodes):
                parent_node.child_nodes[index] = new_node
    def __call__(self,df):
        # 对main_node的封装
        return self.main_node(df)
    def __str__(self) -> str:
        # 对main_node的封装
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
        self.data=[] # 存储所有dataframe
        self.population=[] # 存储目前的树
        self.maxsize=maxsize
        self.names=XTree().names
        for i in range(pop_num): # 生成pop_num个树
            a=XTree(maxsize=maxsize)
            a.split()
            self.population.append(a)
        for i in self.typelist:
            # 导入data/中的数据
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
        logger.debug("Before crossover:")
        logger.debug("Tree1:"+str(tree1.main_node))
        logger.debug("Tree2:"+str(tree2.main_node))
        path1 = [random.randint(0, len(tree1.main_node.child_nodes) - 1)]
        path2 = [random.randint(0, len(tree2.main_node.child_nodes) - 1)]
        #开始随机选择两个树中某个子节点
        node1 = tree1.find_node(path1)
        while node1 and node1.capacity > 0 and random.random() > 0.6: # 如果不是值节点且概率判定通过则目前节点变为目前节点的子节点
            idx = random.randint(0, len(node1.child_nodes) - 1)
            path1.append(idx)
            node1 = node1.child_nodes[idx]

        node2 = tree2.find_node(path2)
        while node2 and node2.capacity > 0 and random.random() > 0.6: # 如果不是值节点且概率判定通过则目前节点变为目前节点的子节点
            idx = random.randint(0, len(node2.child_nodes) - 1)
            path2.append(idx)
            node2 = node2.child_nodes[idx]

        c_node1 = copy.deepcopy(node1) # 深度复制防止引用问题
        c_node2 = copy.deepcopy(node2) # 深度复制防止引用问题

        tree1.replace_node(path1, c_node2) # 替换节点
        tree2.replace_node(path2, c_node1) # 替换节点

        logger.debug("After crossover:")
        logger.debug("Tree1:"+str(tree1.main_node))
        logger.debug("Tree2:"+str(tree2.main_node))

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
        random_data=random.sample(self.data,int(0.4*len(self.data)))
        random_l=len(random_data)
        for  i in range(l):
            # 对每个树计算corr
            corr=pd.DataFrame()
            logger.debug(str(self.population[i]))
            pop=self.population[i]
            # 先将所有corr加起来平均
            for df in random_data:
                df['a']=pop(df)
                df=df[['a','expect1']]
                
                if len(corr)==0:
                    corr=df.corr()
                else:
                    corr=corr+df.corr()
            corr=np.abs(corr.loc['expect1','a']/random_l) #取绝对值
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
        #同样的操作，用于最后一次calculation，区别是计算全品种的平均并且不取绝对值
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
        logger.debug(f'before mutation:{tree}')
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
        
        logger.debug("After mutation:"+str(tree))
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
        # 生成概率分布
        selection_probs = [f / total_fitness for f in fitness]

        new_population = []
        new_population.append(copy.deepcopy(self.population[np.argmax(fitness)])) # 把这一轮适应度最高的直接晋级到下一轮
        while len(new_population) < len(population): #轮盘赌法每次随机抓取两个parent交叉生成子代
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
        # 计算适应度
        fitness=self.calculate_fitness()
        print(fitness)
        best=np.max(fitness)
        best_tree=self.population[np.argmax(fitness)]
        logger.info(f'Best fitness is {best}, best tree is {best_tree}')
        #交叉
        new_population = self.crossover_population_with_selection(self.population,fitness)
        self.population=new_population
        #开始变异
        idx=list(range(1,len(self.population)))
        indexes=random.sample(idx,int(0.1*len(self.population)))
        for i in indexes:
            self.population[i]=self.mutate(self.population[i])
        return best, best_tree
    def run(self,generation=10):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        """
        loop circulation. The main interface of genetic_algorithm.

        Args:
            generation (int, optional): the num of loops. Defaults to 10.
        """
        for i in range(generation):
            try:
                self.loop()
                # if m_fit>0.15:
                #     break
                print(f' generation {i} of {generation} Completed')
            except KeyboardInterrupt:#如果按ctrl C，程序会提前退出迭代，直接计算目前状态并输出
                print("Recieve keyboard interruption, the loop is stopped and begins to dump the current result to factor/auto/. Please wait a while for the program to be appropriately terminated.")
                break
        fitness = self.calculate_all_fitness()
        df=[[x,_] for _, x in sorted(zip(fitness, self.population), reverse=True,key=lambda x: x[0])]
        df=pd.DataFrame(df,columns=['factor','fitness'])
        df['abs']=df['fitness'].apply(np.abs)
        df=df.sort_values(by="abs",ascending=False)
        df=df.drop_duplicates(subset=['factor'])
        df.to_excel(f'factor/auto/auto_factor_pop{len(self.population)}_depth{self.maxsize}_generation{generation}.xlsx')
        
        
if __name__=='__main__':
    g=genetic_algorithm(1000,maxsize=6)
    t=time.time()
    g.run(15)
    print(time.time()-t)


# df=pd.DataFrame()
# df.to_excel('factor/auto/test,.xlsx')
# a=np.zeros(10)
# b=np.nan_to_num(np.corrcoef(a,a))
# print(b)

a=pd.read_csv('data/CU_daily.csv')
