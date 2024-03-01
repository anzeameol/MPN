import os
from dataset import MyDataset
from model import getFeature
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = ['train', 'varify', 'test']
select = 2
outFolder = os.path.join(os.path.abspath('..'), 'output')
npoint = 160
batchsize = 1

# 获取已经训练好的特征提取模型
feature = getFeature()
feature.load_state_dict(torch.load('%s/get_feature.pth' % outFolder))
feature.eval()

# 获取数据集
testDatasetFolder = os.path.join(os.path.abspath('..'), 'dataset', dataset[select])
testDataset = MyDataset(testDatasetFolder, npoint)
testDataloader = torch.utils.data.DataLoader(
    testDataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)

# 将数据集中的数据进行特征提取，并拼接
x = torch.tensor([])
y = torch.tensor([])
label = []
for data in testDataloader:
    points, target = data
    points = points.transpose(2, 1)
    feat = feature(points)
    feat = feat.detach()
    if x.shape[0] == 0:
        x = feat
        y = target
    else:
        x = torch.concat([x, feat], dim=0)
        y = torch.concat([y, target], dim=0)
    label.append(testDataset.classes[target])

# 标准化
scaler = StandardScaler()
scaler.fit(x)
x_std = scaler.transform(x)

# PCA
pca = PCA(n_components=2)  # Retain 2 principal components
x_pca = pca.fit_transform(x_std)

# 可视化
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA--' + dataset[select])
plt.legend(handles=scatter.legend_elements()[0], labels=label, title='Target Class')
plt.show()
