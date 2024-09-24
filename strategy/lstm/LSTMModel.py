 
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import torch.utils.data.dataset as Dataset
import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import pandas as pd
import torch
from torch import nn
import datetime
import time
import os
EPOCH=400
PATH="strategy/lstm/par/CU.pth"
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader
import torch.nn.functional as F
from main import DNN,ValidateData
warnings.simplefilter(action='ignore', category=FutureWarning)

PATH="strategy/lstm/par/CU.pth"
model=DNN(8,1)
model.load_state_dict(torch.load(PATH))
model.eval()
validatedata = ValidateData()
validateloader = DataLoader.DataLoader(dataset=validatedata,batch_size=1,shuffle=True,num_workers=0)
for X,y in validateloader:
    print(f"preidct:{model(X)},real:{y}")
    