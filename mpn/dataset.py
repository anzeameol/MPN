import os
import numpy as np
import random
from torch.utils.data import Dataset

'''
数据预处理：让每个数据中点个数一致
保证张量的维数相同
使用随机剔除点的方法
'''
def proceeding(data, npoint):
    gap = len(data) - npoint
    assert gap > 0
    while gap > 0:
        data = np.delete(data, random.randint(0, len(data) - 1), 0)
        gap = gap - 1
    return data


'''
继承torch.utils.data.Dataset的类，用于处理自己的数据集
需重写__init__和__getitem__和__len__方法
'''
class MyDataset(Dataset):
    def __init__(self,
                 root,
                 npoint
                 ):
        '''
        分析自己制作的数据集的格式
        标签是文件夹名称
        '''
        self.root = root  # 根目录
        self.classes = os.listdir(self.root)  # 类别
        self.datapath = []  # 每个数据的路径
        self.label = []  # 每个数据的标签
        self.npoint = npoint
        for i,label in enumerate(self.classes):
            path = os.path.join(self.root, label)
            dataname = os.listdir(path)
            for name in dataname:
                self.datapath.append(os.path.join(path, name))
                self.label.append(i)

    def __getitem__(self, index):
        data = np.loadtxt(self.datapath[index], dtype='float32', comments='//', delimiter=' ', skiprows=2,usecols=[0, 1, 2])
        data[:,2] *= -1
        data = proceeding(data,self.npoint)  # 数据预处理
        return data, self.label[index]

    def __len__(self):
        return len(self.label)

train_dataset = MyDataset(root,npoint)
