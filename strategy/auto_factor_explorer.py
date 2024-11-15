import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("strategy/")
import os
import multiprocessing
from utils.BackTestEngine import *
from utils.auto_utils import *

# genetic_alorithm sample script

if __name__=='__main__':
    # genetic_algotithm(pop_num,maxsize)
    # pop_num is the number of operators during the iteration. 
    # maxsize controls the initialized depth of each operator.
    g=genetic_algorithm(10000,maxsize=8)
    t=time.time()
    # generation is the number of iterations. 
    g.run(generation=200)
    print(time.time()-t)


# f=FunctionPool()
# df=pd.read_excel('data/CU_daily.xlsx')
# print(df[['expect1','d_position5','skew_position20','vol_kurt126']].corr())


