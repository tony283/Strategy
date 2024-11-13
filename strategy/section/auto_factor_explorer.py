import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("strategy/")
import os
import multiprocessing
from utils.BackTestEngine import *
from utils.auto_utils import *


if __name__=='__main__':
    g=genetic_algorithm(200,maxsize=8)
    t=time.time()
    g.run(generation=200)
    print(time.time()-t)







