import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

'''
变换矩阵
'''
class TransformNet(nn.Module):
    def __init__(self, k):
        super(TransformNet, self).__init__()  # 初始化Module类
        self.k = k

        # 卷积层
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # 线性回归层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        # batch归一化
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]

        # 神经网络结构
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]  # 最大池化

        x = x.view(-1, 1024)  # 重构张量维度
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        '''
        np.eye(n)：生成n阶对角阵
        flatten：展平
        astype：转换数据类型
        view：重构张量维度
        repeat：通过重复排列增加维度
        torch.autograd.Variable：自动计算梯度（现在已被废弃）
        '''
        iden = torch.autograd.Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,
                                                                                                           self.k * self.k).repeat(
            batchsize, 1)  # 偏置矩阵
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)

        return x


'''
获取特征
'''
class getFeature(nn.Module):
    def __init__(self):
        super(getFeature, self).__init__()

        # transformNet
        self.TNet1 = TransformNet(3)
        self.TNet2 = TransformNet(64)

        # 卷积层
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # batch归一化
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):

        # 神经网络结构
        trans = self.TNet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)  # 矩阵乘法，即对x进行变换
        x = x.transpose(2, 1)

        # x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = self.TNet2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        # x = F.relu(self.conv2(x))
        # x = self.conv3(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x


'''
继承nn.Module的一个类，定义自己的神经网络
需重写__init__和forward方法
分类，得到各个类别的分数
'''
class Classification(nn.Module):
    def __init__(self, k=4):
        super(Classification, self).__init__()

        # 获取特征
        self.feature = getFeature()

        # 线性回归层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)  # k为类别个数

        # dropout层
        self.dropout = nn.Dropout(p=0.3)

        # batch归一化
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # relu激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 神经网络结构
        x = self.feature(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.dropout(self.fc2(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # 对得分使用softmax函数
