import math
import os
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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
class DNN(nn.Module):
    def __init__(self, n_input, n_output):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(n_input, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256,64)
        self.layer5 = nn.Linear(64,n_output)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.sigmoid(self.layer4(x))
        
        return self.layer5(x)


class TrainData(Dataset.Dataset):
    def __init__(self):
        self.data :pd.DataFrame = pd.read_excel("data/DQN/onedata.xlsx")
        self.data = self.data.iloc[:500]
        self.factors = ["profitall63","sigma5","sigma20","sigma63","vol","open_interest","profit1","profit2","profit3","profit4","profit5"]

        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.tensor(self.data.iloc[index][self.factors], dtype=torch.float64)
        label = F.softmax(torch.tensor([self.data.iloc[index]["R"]], dtype=torch.float64),dim=0)
        return data, label
class ValidateData(Dataset.Dataset):
    def __init__(self):
        self.data :pd.DataFrame = pd.read_excel("data/DQN/onedata.xlsx")
        self.data = self.data.iloc[-500:]
        self.factors = ["profitall63","sigma5","sigma20","sigma63","vol","open_interest","profit1","profit2","profit3","profit4","profit5"]
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.tensor(self.data.iloc[index][self.factors], dtype=torch.float64)
        label = F.softmax(torch.tensor([self.data.iloc[index]["R"]], dtype=torch.float64),dim=0)
        return data, label
    
    
    
    
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)    







###############################################
traindata =TrainData()
validatedata = ValidateData()
dataloader =DataLoader.DataLoader(dataset=traindata,batch_size=64,shuffle=True,num_workers=0)
validateloader = DataLoader.DataLoader(dataset=validatedata,batch_size=64,shuffle=True,num_workers=0)
model=DNN(11,1).to(device)

loss_fn = nn.MSELoss()
optimizer= optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)


'''定义训练函数'''
def train(dataloader,model,loss_fn,optimizer):
    # 设置模型为训练模式
    model.train()
    # 记录优化次数
    num=1
    my_loss = 0
    # 遍历数据加载器中的每一个数据批次。
    for X,y in dataloader:
        X,y=X.to(device),y.to(device)
        # 自动初始化权值w
        pred=model.forward(X)
        # print(pred)
        loss=loss_fn(pred,y) # 计算损失值
        # 将优化器的梯度缓存清零
        optimizer.zero_grad()
        # 执行反向传播计算梯度
        loss.backward()
        optimizer.step()
        loss_value=loss.item()
        # print(f'loss:{loss_value},[numbes]:{num}')
        # loss.backward()
        my_loss+=loss_value
        num+=1
    return my_loss

EPOCH = 20
PATH = "data/DQN/ADNN.pt"
if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH))
    model.eval()
train_loss_list = []
validate_loss_list = []
loss_df = pd.DataFrame(columns=["loss"])
for i in range(EPOCH):
    m_loss = train(dataloader=traindata,model=model,loss_fn=loss_fn,optimizer=optimizer)
    print(f'EPOCH{i}:loss is {m_loss}')
    train_loss_list.append(m_loss)
    torch.save(model.state_dict(),PATH)
    #开始测试
    with torch.no_grad():  # 确保不会进行反向传播计算梯度，节省内存和计算资源
        model.eval()
        validate_loss = 0
        for X,y in validateloader:
            X,y=X.to(device),y.to(device)
            test_outputs = model(X)  # 前向传播获取测试集的预测结果
            validate_loss += loss_fn(test_outputs, y)  # 计算测试集上的损失值
        print(f'Test Loss: {validate_loss.item()}')  # 打印测试损失信息
        validate_loss_list.append(validate_loss)
loss_df["train_loss"]=train_loss_list
loss_df["validate_loss"]=validate_loss_list
try:
    loss_log = pd.read_excel("data/DQN/loss.xlsx",index=None)
    loss_log = pd.concat([loss_log,loss_df])
    loss_log.to_excel("data/DQN/loss.xlsx")
except:
    loss_df.to_excel("data/DQN/loss.xlsx")
    

    
