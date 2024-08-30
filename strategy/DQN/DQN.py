import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch.utils.data.dataset as Dataset
from itertools import count
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.utils.data.dataloader as DataLoader
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256,n_actions)
        self.layer5 = nn.Softmax(dim=0)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        
        return self.layer5(x)


class Datas(Dataset.Dataset):
    def __init__(self):
        self.data :pd.DataFrame = pd.read_excel("data/DQN/alldata.xlsx")
        self.newlist = ['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'BU', 'FG',  'L', 'MA', 'PP', 'RU',
           'TA', 'V', 'A', 'C', 'CF', 'M', 'OI', 'RM', 'SR', 'Y', 'JD',  'B', 'P', 'AL', 'CU', 'PB', 'ZN']
        self.factor2 = [i+"Current" for i in  self.newlist]
        self.factors = self.newlist.copy()
        self.factors.extend(self.factor2)
        self.reward = [i+"Reward" for i in  self.newlist]
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.tensor(self.data[self.factors].iloc[index], dtype=torch.float32)
        label = F.softmax(torch.tensor(self.data[self.reward].iloc[index], dtype=torch.float32),dim=0)
        return data, label
dataset =Datas()
dataloader =DataLoader.DataLoader(dataset=dataset,batch_size=1,shuffle=True,num_workers=0)
for data, label in dataloader:
    print(f"data is {data},result is{label}")

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )    
