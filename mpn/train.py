import argparse
import random
import os
from dataset import MyDataset
from model import Classification
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''
设置参数
'''
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=8, help='input batch size')
parser.add_argument(
    '--npoint', type=int, default=160, help='input number of points')
parser.add_argument(
    '--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument(
    '--nepoch', type=int, default=20, help='number of epochs to train for')

opt = parser.parse_args()
# print(opt)

outFolder = os.path.join(os.path.abspath('..'), 'output')
trainDatasetFolder = os.path.join(os.path.abspath('..'), 'dataset', 'train')
varifyDatasetFolder = os.path.join(os.path.abspath('..'), 'dataset', 'varify')
testDatasetFolder = os.path.join(os.path.abspath('..'), 'dataset', 'test')

'''
训练：在命令行输入如下语句
'''
# python train.py --batchSize 8 --npoint 160 --workers 0 --nepoch 10

'''
设置随机种子，方便下次复现实验结果
'''
opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

'''
实例化dataset子类
'''
trainDataset = MyDataset(trainDatasetFolder, npoint=opt.npoint)
varifyDataset = MyDataset(varifyDatasetFolder,npoint=opt.npoint)
testDataset = MyDataset(testDatasetFolder, npoint=opt.npoint)

'''
实例化dataloader类
'''
trainDataloader = torch.utils.data.DataLoader(
    trainDataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

varifyDataloader = torch.utils.data.DataLoader(
    varifyDataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testDataloader = torch.utils.data.DataLoader(
    testDataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

'''
设置输出目录
'''
try:
    os.makedirs(outFolder)
except OSError:
    pass

nclasses = len(trainDataset.classes)
classifier = Classification(nclasses)  # 神经网络模块

'''
优化器
'''
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))  # Adam算法优化
# optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 学习率调整优化
classifier.cuda()  # 将模型加载到gpu上

'''
训练
'''
nbatch = len(trainDataset) / opt.batchSize  # 多少个批次
varify_step = len(trainDataset) / len(varifyDataset)  # 什么时候使用验证集
nvarify = nbatch / varify_step
blue = lambda x: '\033[94m' + x + '\033[0m'  # 蓝色

for epoch in range(opt.nepoch):
    for i, data in enumerate(trainDataloader):
        points, target = data  # 获取打包的数据和标签
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()  # 使用gpu运算
        optimizer.zero_grad()  # 将梯度置0，防止与其他batch混合运算
        classifier = classifier.train()  # 训练
        pred = classifier(points)  # 得到每个类别的分数
        loss = torch.nn.functional.nll_loss(pred, target)  # 计算loss
        loss.backward()  # 反向传递
        optimizer.step()
        pred_choice = pred.data.max(1)[1]  # 预测的类别
        correct = pred_choice.eq(target.data).cpu().sum()  # 正确的个数
        print('[epoch%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, nbatch, loss.item(), correct.item() / float(opt.batchSize)))

        # 验证
        if (i+1) % varify_step == 0:
            j, data = next(enumerate(varifyDataloader))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()  # 验证，不启用Batch Normalization和Dropout
            pred = classifier(points)
            loss = torch.nn.functional.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[epoch%d] %s loss: %f accuracy: %f' % (
                epoch, blue('varify'), loss.item(), correct.item() / float(opt.batchSize)))

    # scheduler.step()  # 学习率调整

torch.save(classifier.state_dict(), '%s/cls_model.pth' % outFolder)  # 保存模型参数
torch.save(classifier.feature.state_dict(), '%s/get_feature.pth' % outFolder)

'''
测试
使用tqdm库（迭代进度条库）
'''
# total_correct = 0
# total_testset = 0
# for i, data in tqdm(enumerate(testDataloader, 0)):
#     points, target = data
#     points = points.transpose(2, 1)
#     points, target = points.cuda(), target.cuda()
#     classifier = classifier.eval()
#     pred = classifier(points)
#     pred_choice = pred.data.max(1)[1]
#     correct = pred_choice.eq(target.data).cpu().sum()
#     total_correct += correct.item()
#     total_testset += points.size()[0]
#
# print("final accuracy: {}".format(total_correct / float(total_testset)))


