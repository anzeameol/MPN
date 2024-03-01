<center><font size = 7>Some Notes</font></center>
<p align='right'>by Nemo</p>
<p align='right'>2023.8.12</p>

### 使用argparse传参
#### code:
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
#...more option...

```
其他参数：
- required：是否必需
- nargs：多少个参数
- action: action关键字默认状态有两种，store_true和store_false。若输入命令时指定了此参数，则根据选择的是store_true\store_false将此参数设置为true\false，如果未指定则取相反的

opt = parser.parse_args()
print(opt)

> usage: 打开命令行输入：train.py [-h] [--batchSize BATCHSIZE]
> //-h表示--help
> e.g. train.py --batchSize 5

#### 获取参数
```python
opt.batchSize
```

### python
#### super函数
通常情况下，我们在子类中定义了和父类同名的方法，那么子类的方法就会覆盖父类的方法。而super关键字实现了对父类方法的改写(增加了功能，增加的功能写在子类中，父类方法中原来的功能得以保留)。也可以说，super关键字帮助我们实现了在子类中调用父类的方法。  
比如a类继承了torch.nn.Module类，想调用torch.nn.Module类的__init__方法，就可以使用super(a,self).__init__()

### pytorch
#### torch.manual_seed()：
设置CPU生成随机数的种子，方便下次复现实验结果

#### torch.nn.Module类：
使用Module类来自定义模型  
我们在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法。但有一些注意技巧：
1. 一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，当然我也可以吧不具有参数的层也放在里面；
2. 一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)可放在构造函数中，也可不放在构造函数中，如果不放在构造函数__init__里面，则在forward方法里面可以使用nn.functional来代替
3. forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。