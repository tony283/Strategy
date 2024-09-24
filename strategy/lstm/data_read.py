# 加载数据集并划分数据集和测试集
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

 
 
# 自定义一个数据集类
class TrainData(Dataset):
    def __init__(self):
        self.data = pd.read_excel(f"strategy/lstm/train_data/CU.xlsx")
        self.data=self.data.iloc[:int(len(self.data)*0.9)]

 
        # 提取特征和目标变量
        self.X = self.data[["profit","profit20","profit63","profit126","profit252","macd1","macd2","macd3"]]  # 除了目标剩下的都是特征
        self.Y = self.data['expect']  # 目标
 
    
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        # 从self.X中选择索引为idx的行，并将其赋值给X_sample
        X_sample = self.X.iloc[idx]
        Y_sample = self.Y.iloc[idx]
 
        # 特征数据和标签数据都被转换为float32类型的PyTorch张量。
        sample = {'X': torch.tensor(X_sample.values, dtype=torch.float32),
                  'Y': torch.tensor(Y_sample, dtype=torch.float32)}
        return sample
 
 
class ValidateData(Dataset):
    def __init__(self):
        self.data = pd.read_excel(f"strategy/lstm/train_data/CU.xlsx")
        self.data=self.data.iloc[-int(len(self.data)*0.1):]

 
        # 提取特征和目标变量
        self.X = self.data[["profit","profit20","profit63","profit126","profit252","macd1","macd2","macd3"]]  # 除了目标剩下的都是特征
        self.Y = self.data['expect']  # 目标
 
    
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        # 从self.X中选择索引为idx的行，并将其赋值给X_sample
        X_sample = self.X.iloc[idx]
        Y_sample = self.Y.iloc[idx]
 
        # 特征数据和标签数据都被转换为float32类型的PyTorch张量。
        sample = {'X': torch.tensor(X_sample.values, dtype=torch.float32),
                  'Y': torch.tensor(Y_sample, dtype=torch.float32)}
        return sample