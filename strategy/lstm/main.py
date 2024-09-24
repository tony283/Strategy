import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import sys
sys.path.append("strategy/lstm/")
from model import LSTMModel
from data_read import TrainData,ValidateData
import torch.nn as nn
import torch.optim as optim
import os
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)    







###############################################
traindata =TrainData()
validatedata = ValidateData()
dataloader =DataLoader.DataLoader(dataset=traindata,batch_size=128,shuffle=True,num_workers=0)
validateloader = DataLoader.DataLoader(dataset=validatedata,batch_size=128,shuffle=True,num_workers=0)
model=LSTMModel(8,64,1).to(device)
print(device)
loss_fn = nn.MSELoss()
optimizer= optim.Adam(model.parameters(), lr=0.001)


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
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        loss_value=loss.item()
        # print(f'loss:{loss_value},[numbes]:{num}')
        # loss.backward()
        my_loss+=loss_value
        num+=1
    return my_loss



EPOCH = 100
PATH = "strategy/lstm/par/CU.pt"
BACKPATH = "strategy/lstm/par/CU"
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
            # print(f"Predict is {test_outputs},real value is {y}")
            validate_loss += loss_fn(test_outputs, y).item()  # 计算测试集上的损失值
        print(f'Test Loss: {validate_loss}')  # 打印测试损失信息
        validate_loss_list.append(validate_loss)
loss_df["train_loss"]=[t for t in  train_loss_list]
loss_df["validate_loss"]=[t for t in  validate_loss_list]
try:
    loss_log = pd.read_excel("data/DQN/loss.xlsx",index=None)
    loss_log = pd.concat([loss_log,loss_df])
    loss_log.to_excel("data/DQN/loss.xlsx")
except:
    loss_df.to_excel("data/DQN/loss.xlsx")
    

    
