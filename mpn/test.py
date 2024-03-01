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
outFolder = os.path.join(os.path.abspath('..'), 'output')
classifier = Classification()
classifier.load_state_dict(torch.load('%s/cls_model.pth' % outFolder))
select = 2
outFolder = os.path.join(os.path.abspath('..'), 'output')
npoint = 160
batchsize = 1

dataset = ['train', 'varify', 'test']
testDatasetFolder = os.path.join(os.path.abspath('..'), 'dataset', dataset[select])
testDataset = MyDataset(testDatasetFolder, npoint)
testDataloader = torch.utils.data.DataLoader(
    testDataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)
classifier.cuda()
total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(testDataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred = classifier(points)
    pred_choice = pred.data.max(1)[1]
    print(pred_choice,target)
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy: {}".format(total_correct / float(total_testset)))