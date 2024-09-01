import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  #输出为True，则安装无误
print("使用gpu数量为：", torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
try:
    print(torch.cuda.get_device_name(1))
except:
    print("并未使用其他显卡")