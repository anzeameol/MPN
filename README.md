<center><font size = 5>fdu computer graphic projects</font></center>
<p align='right'>by Nemo</p>

# Introduction
自制四分类点云数据集，使用CloudCompare生成立方体，球体，圆柱体，圆锥四种点云数据

参考pointnet框架，实现了一个深度学习网络，加入自己的数据集读取和预处理步骤，实现分类任务

进行消融实验，取消数据的归一化、标准化、学习率调整，更改优化器、学习率、网络结构，观察影响；使用PCA将特征降维并可视化，观察4个类别对应的特征，验证网络提取点云特征的可靠性