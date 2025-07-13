1.这里构建了一个非常非常非常基础的卷积神经网络模型，从认识 nn.Module 开始，了解如何简便地运用 nn.Module 来实现模型定义

2.代码虽然简短，但实现了深度学习最核心的梯度下降优化过程（我们所说的梯度本质上是计算损失函数对于每个参数的偏导数）

3.如果你想要更改数据的尺寸，不要忘了同步修改全连接层的参数设定，具体计算方式在注释中有列出

----------
By the way 提一下batch_size, batch 意思是“批次”，在创建迭代器 DataLoader 时设置，用于将数据集分成多个批次，表示每一次传递给模型用于训练的数据样本的数量。于是我们很容易想象到：
   如果batch_size设置得很小，则模型需要训练很多轮，收敛速度就会偏慢，但这较适用于内存有限的设备；
   如果batch_size设置得太大，计算效率提高了，但但模型可能不一定能够收敛到最优解
   通常像64，128这样较好
----------


# # nn.Module是所有神经网络的基类，提供神经网络所有基础功能（参数管理、计算图构建、GPU加速支持、模型保存/加载）
# # 任何自定义网络都必须继承nn.Module

# class MyModel(nn.Module):
#     def __init__(self):
#         super().__init__()#必须调用
#         self.layer = nn.Linear(10,5)
#
#     def forward(self, x):#也是必须的
#         return self.layer(x)


# # nn.Module会自动跟踪所有通过nn.Parameter注册的参数
# class ParaModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(3,5))#随机生成一个（3，5）的矩阵
#         self.bias = nn.Parameter(torch.zeros(3))#三个0
#         self.dummy = torch.ones(3) #此时普通张量就不会被优化器识别，不会输出
# model = ParaModel()
# print(list(model.parameters()))
# # model.parameters()获取所有可训练参数
# print(list(model.named_parameters()))
# # 加了named，会返回(name,param)的键值对


----------开----------始----------咯----------


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# 可嵌套组合各种nn.Module
# 以下定义了一个卷积神经网络，适用于小尺寸图像分类（如MINIST、CIFAR-10）
class MegaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            # 接受3通道的图片，利用16个3x3卷积核提取特征，假设输入尺寸（batch_size,3,224,224）
            nn.Conv2d(3,16,3),
            # 输出尺寸（batch_size,16,222,222) (224-3+0)/1+1=222
            nn.ReLU(),
            # 下采样，以缩小尺寸，减少计算量
            nn.MaxPool2d(2)
            # 输出维度（batch_size,16,111,111) 2x2窗口，步长为2,222/2=111，将学到的特征映射到10个类别上
        )
        self.fc = nn.Linear(16*111*111,10)# 全连接层

    def forward(self,x):
        x = self.conv_net(x)  # 获取形状（batch_size,16,111,111）
        # x = x.view(x.size(0), -1)  # 展平（batch_size,197136）16x111x111=197136, -1表示自动计算该维度的大小
        # 现代pytorch展平操作推荐使用nn.Flatten()
        self.flatten = nn.Flatten()
        x = self.flatten(x)
        x = self.fc(x)  #（batch_size,10）
        return x

# 创建一个dataloader实例用于迭代

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集
data = torch.randn(100, 3, 224, 224)
labels = torch.randint(0, 10, (100,))
dataset = MyDataset(data, labels)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MegaModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 训练循环
    for image, labels in dataloader:
        outputs = model(image)  # 前向传播完成 （batch_size,num_classes）
        loss = criterion(outputs, labels)
        # labels means 真实标签(batch_size)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度,梯度会存入各个参数的.grad属性中
        optimizer.step()  # 根据梯度更新模型参数
        running_loss += loss.item()  # 累计损失
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)  # 平均损失=累计损失/批次总数
    epoch_accuracy = 100.0 * correct / total  # 正确预测的样本数/总样本数
    print(f'Epoch[{epoch+1}/{num_epochs}],Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    # loss保留四位小数，accu保留两位小数并加上%


----------运行此代码，你会看到，总共十轮训练，每训练一次，就会显示本轮训练的损失和准确率----------
