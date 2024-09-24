# 加载数据集并划分数据集和测试集
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
# 自定义一个数据集类
class CustomDataset(Dataset):
    def __init__(self,duration):
        self.data = pd.read_excel(f"strategy/lstm/train_data/CU.xlsx")
        self.data=self.data.iloc[:int(len(self.data)*0.9)]
        self.duration = duration

 
        # 提取特征和目标变量
        self.X = self.data[["profit","profit20","profit63","profit126","profit252","macd1","macd2","macd3"]]  # 除了目标剩下的都是特征
        self.Y=self.data["expect"]
 
    
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        # 从self.X中选择索引为idx的行，并将其赋值给X_sample
        X_sample = self.X.iloc[idx:idx+self.duration]
        Y_sample = self.Y.iloc[idx+self.duration]
 
        # 特征数据和标签数据都被转换为float32类型的PyTorch张量。
        sample = {'X': torch.tensor(X_sample.values, dtype=torch.float32),
                  'Y': torch.tensor(Y_sample, dtype=torch.float32)}
 
        return sample


# 加载数据集并划分数据集和测试集
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

 
 

 
 
def load_data():
    # 读取CSV文件
 
    # 创建数据集实例
    custom_dataset = CustomDataset(14)
 
    # 划分训练集和测试集
    train_size = int(0.9 * len(custom_dataset))  # 训练集占比90%
    test_size = len(custom_dataset) - train_size  # 测试集占比10%
    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])  # 按照比例随机划分
 
    # 创建数据加载器
    # batch_size参数用于指定每个批次（batch）中包含的样本数量。
    # 通常情况下，较大的batch_size可以加快训练速度，但可能会占用更多的内存资源。
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
    # 检查数据加载器
    for batch in train_loader:
        print(batch['X'].shape, batch['Y'].shape)
        break
 
    return train_loader, test_loader, custom_dataset
