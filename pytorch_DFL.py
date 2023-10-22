import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
from torch.utils.data import random_split
'''
选出一个领袖设备或节点，作为本次训练的中心节点。
在训练阶段，领袖节点将模型参数广播给周围的设备或节点，
这些设备或节点根据本地数据计算梯度并返回给领袖节点，
领袖节点根据所有梯度更新模型参数。
这个过程会循环迭代，直到模型收敛为止。
'''

# 定义一个设备或节点类，其中包含数据集、模型、优化器等信息
class Device:
    def __init__(self, device_id, train_data, test_data):
        self.id = device_id
        self.train_data = train_data
        self.test_data = test_data
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            # nn.ReLU(True),
            nn.Linear(128, 10),
            nn.Sigmoid(),
            # nn.ReLU(True)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def forward(self, x):
        x = self.model(x)
        return x


    def receive_gradients(self, gradients):
        return gradients


# 定义一个简单的选举算法，选出一个领袖设备或节点
def elect_leader(devices):
    leaders = []
    for device in devices:
        if random.random() < 0.5:
            leaders.append(device)
    if len(leaders) == 0:
        return None
    else:
        return max(leaders, key=lambda x: x.id)


# 定义一个用于计算梯度的函数
def calculate_gradients(device, model):
    train_loss = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    data_loader = DataLoader(device.train_data, batch_size=64, shuffle=True)
    device_gradients = []

    # if num_round % 5 == 0:
    #     device.optimizer.param_groups[0]['lr'] *= 0.9

    for inputs, labels in data_loader:
        # 前向传播
        outputs = model(inputs.view(-1, 784))
        # print(outputs)
        # print(labels)
        loss = loss_fn(outputs, labels)
        # 反向传播
        device.optimizer.zero_grad()
        loss.backward()
        device.optimizer.step()
        device_gradients.append(copy.deepcopy(device.model.state_dict()))
        train_loss += loss.item()  # 所有批次损失的和
        _, predicted = torch.max(outputs.data, 1)  # 找到每个输出张量中的最大值及其索引
        correct += (predicted == labels).sum().item()
        return device_gradients, round(train_loss / len(data_loader), 4), round(correct*100/len(data_loader), 2)

def train_device(device, local_epochs):
    losses = []  # 记录训练集损失
    acc = []
    for i in range(local_epochs):
        # print(device.model.state_dict())
        device.model.load_state_dict(device.model.state_dict())  # 加载终端设备的模型参数
        gradients, train_loss, train_acc = calculate_gradients(device, device.model)  # 在本地训练
        losses.append(train_loss)  # 所有样本平均损失
        acc.append(train_acc)

    # print("train_loss: ", losses)
    # print("train_acc: ", acc)
    return gradients

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
    # 读入数据
    train_set = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # 拆分数据，拆成3份
    train_size = len(train_set)
    subset_sizes = [train_size // 3, train_size // 3, train_size - 2 * (train_size // 3)]
    train_subset1, train_subset2, train_subset3 = random_split(train_set, subset_sizes)

    test_size = len(test_set)
    subset_sizes = [test_size // 3, test_size // 3, test_size - 2 * (test_size // 3)]
    test_subset1, test_subset2, test_subset3 = random_split(test_set, subset_sizes)

    # 实例化3个设备
    device1 = Device(1, train_subset1, test_subset1)
    device2 = Device(2, train_subset2, test_subset2)
    device3 = Device(3, train_subset3, test_subset3)
    devices = [device1, device2, device3]

    # 开始训练
    for i in range(10):
        gradients_list = []
        for device in devices:
            gradients = train_device(device, 5)
            gradients_list.append(gradients)

        # 聚合模型--平均
        average_gradients = {}
        for key in gradients_list[0][0].keys():
            # print(key)
            average_gradients[key] = torch.mean(torch.stack([gradients[0][key] for gradients in gradients_list]), dim=0)

        # 选择领导设备
        # leader = None
        # while leader is None:
        #     leader = elect_leader(devices)
        # print(leader.id)
        # 更新梯度
        for device in devices:
            device.model.load_state_dict(
                {key: value - 0.01 * average_gradients[key] for key, value in device.model.state_dict().items()})
        #print(leader.model.state_dict())


    # 测试集预测
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # 在测试集上评估模型
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)  # 加载测试集
    correct = 0
    total = 0
    eval_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in test_loader:  # 迭代测试集中的数据批次
            outputs = device3.model(inputs.view(-1, 784))  # device1 的模型
            loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)   # 找到每个输出张量中的最大值及其索引

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 记录误差
            eval_loss += loss.item()

    print('total: %d ' % total)
    print('correct: %d ' % correct)
    print('Accuracy of the network on the test set: %.2f %%' % (100 * correct / total))
    print('test_loss: %.4f ' % (eval_loss / total))
