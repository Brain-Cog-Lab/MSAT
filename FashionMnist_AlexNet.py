# -*- coding: utf-8 -*-            
# Time : 2022/10/22 19:10
# Author : Regulus
# FileName: FashionMnist_AlexNet.py
# Explain: 
# Software: PyCharm

import os
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=256, type=int, help='simulation time')
parser.add_argument('--p', default=1, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=1, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--lateral_inhi', default=True, type=bool, help='LIPooling')
parser.add_argument('--sin_t', default=256, type=int, help='sin timestep')
parser.add_argument('--data_norm', default=True, type=bool, help=' whether use data norm or not')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='AlexNet', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--useSC', action='store_true', default=False, help='use SpikeConfidence')
parser.add_argument('--train_batch', default=200, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=200, type=int, help='batch size for testing')
parser.add_argument('--seed', default=23, type=int, help='seed')
parser.add_argument('--VthHand', default=1, type=float, help='Vth scale, -1 means variable')
parser.add_argument('--useDET', action='store_true', default=False, help='use DET')
parser.add_argument('--useDTT', action='store_true', default=False, help='use DTT')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import time
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import *

# 构建AlexNet网络
class AlexNet(nn.Module):
    # 网络初始化
    def __init__(self):
        super(AlexNet, self).__init__()
        # 在容器中构建AlexNet网络
        self.features = nn.Sequential(
            # 使用11x11的卷积核捕捉对象，大窗口
            # s=4，减少输出的高度和宽度
            nn.Conv2d(1, 96, (11, 11), stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            # 使用小的卷积核，填充为2，来保持输出的大小不变
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            # 连续三次卷积，前两次卷积都会增加输出的通道数
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
        self.classies = nn.Sequential(
            # 使用dropout技术防止过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(),  # 防止过拟合
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(),
            # 输出层 ，使用的时Fashion-Minist 数据集，分类总数为10
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        features = self.features(x)
        out = self.classies(features.view(x.shape[0], -1))
        return out

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate_snn(test_iter, snn, net, device=None, duration=50):
    t = 1
    folder_path = ""
    while True:
        folder_path = "./result_conversion_{}/parameters_group{}/snn_VthHand{}_useDET_{}_useDTT_{}_useSC_{}".format(
            args.model_name, t, args.VthHand, args.useDET, args.useDTT, args.useSC)
        if os.path.exists(folder_path):
            t += 1
        else:
            os.makedirs(folder_path)
            break
    FolderPath.folder_path = folder_path
    ModelName.name = args.model_name
    accs = []
    total_spikes = []
    num_total_spikes = 0
    snn.eval()
    for ind, (test_x, test_y) in enumerate(tqdm(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        out = 0
        with torch.no_grad():
            clean_mem_spike(snn)
            acc = []
            for t in range(duration):
                out += snn(test_x)
                result = out.max(-1, keepdim=True)[1]
                result = result.squeeze(1).to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)
                if ind == 0:
                    for name, layer in snn.named_modules():
                        if isinstance(layer, SNode):
                            num_total_spikes += layer.all_spike.sum().cpu()
                    total_spikes.append(num_total_spikes / args.batch_size)
        accs.append(np.array(acc))

    if True:
        f = open('{}/result.txt'.format(folder_path), 'w')
        f.write("Setting Arguments.. : {}\n".format(args))
        accs = np.array(accs).mean(axis=0)
        for iii in range(256):
            if iii < 16 or (iii + 1) % 16 == 0:
                f.write("timestep {}:{}\n".format(str(iii+1).zfill(3), accs[iii]))
        f.write("max accs: {}, timestep:{}\n".format(max(accs), np.where(accs == max(accs))))
        f.write("use for latex tabular: & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\n".format(max(accs) * 100, accs[31]* 100,
                                                                                         accs[63]*100, accs[127]*100, accs[255]*100))
        f.write("total spike use for latex tabular: {} & {} & {} & {}".format(total_spikes[31], total_spikes[63], total_spikes[127], total_spikes[255]))
        f.close()
        accs = torch.from_numpy(accs)
        torch.save(accs, "{}/accs.pth".format(folder_path))


if __name__ == '__main__':

    # FashionMnist 数据集分辨率28*28，但是AlexNet网络输入为224*224，所以使用函数将大小扩充成224
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # 这里归一化，因为FashionMnist数据集时灰度图，单通道，所以归一化时是单通道归一化
        transforms.Normalize((0.1307), (0.3081))])

    # 加载数据集
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=data_transforms
    )
    # 构建训练集数据载入器 并提供给定数据集的可迭代对象。
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=data_transforms
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model = AlexNet()
    model.eval()
    device = torch.device("cuda:0")
    net = model.to(device)
    net.load_state_dict(torch.load('AlexNet_model/model.pth', map_location="cuda:0"))

    total = 0.0
    correct = 0.0
    with torch.no_grad():  # 训练集不需要反向传播
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)

            pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
            total += inputs.size(0)
            correct += torch.eq(pred, labels).sum().item()
    print("Accuracy of the network on the 10000 test images:%.2f %%" % (100 * correct / total))

    net1 = deepcopy(net)

    converter = Converter(train_loader, device, args.p, args.lateral_inhi,
                          args.gamma, args.smode, args.VthHand, args.useDET, args.useDTT, args.useSC)
    snn = converter(net)  # use threshold balancing or not
    print(snn)
    converter = Converter(test_loader, device, args.p, args.lateral_inhi,
                          args.gamma, False, args.VthHand, args.useDET, args.useDTT, args.useSC)
    model1 = converter(net1)  # use threshold balancing or not

    evaluate_snn(test_loader, snn, model1, device, duration=args.T)

    # net = AlexNet().train()
    # device = "cuda:0"
    # if args.cuda:
    #     print("use cuda")
    #     cudnn.benchmark = True
    #     net.to(device)
    #
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    #
    # epoch_size = 150
    # base_lr = args.lr
    # criterion = nn.CrossEntropyLoss()  # 定义损失函数：交叉熵
    # acc = []
    # start = time.time()
    # best_acc = 0.0
    # for epoch in range(150):
    #     train_loss = 0.0
    #
    #     # 使用阶梯学习率衰减策略
    #     if epoch in [75, 120]:
    #         tmp_lr = tmp_lr * 0.1
    #         set_lr(optimizer, tmp_lr)
    #
    #     for iter_i, (inputs, labels) in enumerate(train_loader, 0):
    #         # 使用warm-up策略来调整早期的学习率
    #         if not args.no_warm_up:
    #             if epoch < args.wp_epoch:
    #                 tmp_lr = base_lr * pow((iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
    #                 set_lr(optimizer, tmp_lr)
    #
    #             elif epoch == args.wp_epoch and iter_i == 0:
    #                 tmp_lr = base_lr
    #                 set_lr(optimizer, tmp_lr)
    #
    #         # 将数据从train_loader中读出来,一次读取的样本是32个
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels).cuda()
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #
    #     print('[epoch: %d] loss: %.3f' % (epoch + 1, train_loss / len(train_loader)))
    #     lr_1 = optimizer.param_groups[0]['lr']
    #     print("learn_rate:%.15f" % lr_1)
    #
    #     # 由于训练集不需要梯度更新,于是进入测试模式
    #     net.eval()
    #     correct = 0.0
    #     total = 0
    #     with torch.no_grad():  # 训练集不需要反向传播
    #         for inputs, labels in test_loader:
    #             inputs, labels = inputs.cuda(), labels.cuda()
    #             outputs = net(inputs)
    #
    #             pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
    #             total += inputs.size(0)
    #             correct += torch.eq(pred, labels).sum().item()
    #     print("Accuracy of the network on the 10000 test images:%.2f %%" % (100 * correct / total))
    #     if best_acc < (100 * correct / total):
    #         best_acc = 100 * correct / total
    #         print('Saving epoch %d model ..., test acc is %.2f' % (epoch + 1, best_acc))
    #         torch.save(net.state_dict(), './AlexNet_model/model.pth')
    #     acc.append(100 * correct / total)
    #     net.train()
    # end = time.time()
    # print("time:{}".format(end - start))
