import sys
sys.path.append('..')
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import os

from CIFAR100_VGG16 import VGG16
from CIFAR100_ResNet import ResNet20, BasicBlock
import argparse
from utils import *


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=256, type=int, help='simulation time')
parser.add_argument('--p', default=1, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=1, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--lateral_inhi', default=True, type=bool, help='LIPooling')
parser.add_argument('--sin_t', default=256, type=int, help='sin timestep')
parser.add_argument('--data_norm', default=True, type=bool, help=' whether use data norm or not')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--device', default='2', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='vgg16', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--train_batch', default=200, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=200, type=int, help='batch size for testing')
parser.add_argument('--seed', default=23, type=int, help='seed')
parser.add_argument('--VthHand', default=1, type=float, help='Vth scale, -1 means variable')
parser.add_argument('--useDET', action='store_true', default=False, help='use DET')
parser.add_argument('--useDTT', action='store_true', default=False, help='use DTT')
parser.add_argument('--useSC', action='store_true', default=False, help='use SpikeConfidence')
args = parser.parse_args()


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
    snn.eval()
    if args.model_name == "vgg16":
        spike_rate_dict = {
            'relu1': [[] for x in range(8)],
            'relu2': [[] for x in range(8)],
            'relu3': [[] for x in range(8)],
            'relu4': [[] for x in range(8)],
            'relu5': [[] for x in range(8)],
            'relu6': [[] for x in range(8)],
            'relu7': [[] for x in range(8)],
            'relu8': [[] for x in range(8)],
            'relu9': [[] for x in range(8)],
            'relu10': [[] for x in range(8)],
            'relu11': [[] for x in range(8)],
            'relu12': [[] for x in range(8)],
            'relu13': [[] for x in range(8)]
        }
    elif args.model_name == "resnet20":
        spike_rate_dict = {
            'relu1': [[] for x in range(8)],
            'relu2': [[] for x in range(8)],
            'relu3': [[] for x in range(8)],
            'relu4': [[] for x in range(8)],
            'relu5': [[] for x in range(8)],
            'relu6': [[] for x in range(8)],
            'relu7': [[] for x in range(8)],
            'relu8': [[] for x in range(8)],
            'relu9': [[] for x in range(8)],
            'relu10': [[] for x in range(8)],
            'relu11': [[] for x in range(8)],
            'relu12': [[] for x in range(8)],
            'relu13': [[] for x in range(8)],
            'relu14': [[] for x in range(8)],
            'relu15': [[] for x in range(8)],
            'relu16': [[] for x in range(8)],
            'relu17': [[] for x in range(8)],
            'relu18': [[] for x in range(8)],
            'relu19': [[] for x in range(8)],
        }
    # if os.path.exists(r'{}/Vth_timestep.txt'.format(folder_path)): os.remove(r'{}/Vth_timestep.txt'.format(folder_path))
    # if os.path.exists(r'{}/Vmem_timestep.txt'.format(folder_path)): os.remove(r'{}/Vmem_timestep.txt'.format(folder_path))
    # if os.path.exists(r'{}/Vrd_timestep.txt'.format(folder_path)): os.remove(r'{}/Vrd_timestep.txt'.format(folder_path))
    # if os.path.exists(r'{}/Spike_timestep.txt'.format(folder_path)): os.remove(r'{}/Spike_timestep.txt'.format(folder_path))
    # if os.path.exists(r'{}/Maxthreshold_timestep.txt'.format(folder_path)): os.remove(r'{}/Maxthreshold_timestep.txt'.format(folder_path))
    sin_ratio = []
    for ind, (test_x, test_y) in enumerate(tqdm(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        x = deepcopy(test_x)
        out = 0
        with torch.no_grad():
            clean_mem_spike(snn)
            acc = []
            index = 1
            for t in range(duration):
                out += snn(test_x)
                f = open('{}/Vth_timestep.txt'.format(folder_path), 'a+')
                f.write("\n")
                f.close()
                f = open('{}/Vmem_timestep.txt'.format(folder_path), 'a+')
                f.write("\n")
                f.close()
                f = open('{}/Vrd_timestep.txt'.format(folder_path), 'a+')
                f.write("\n")
                f.close()
                f = open('{}/Spike_timestep.txt'.format(folder_path), 'a+')
                f.write("\n")
                f.close()
                f = open('{}/Maxthreshold_timestep.txt'.format(folder_path), 'a+')
                f.write("\n")
                f.close()
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)

                if (t + 1) % 32 == 0:
                    for name, layer in snn.named_modules():
                        if isinstance(layer, SNode):
                            spike_rate_dict['relu' + str(index)][(t + 1) // 32 - 1].append(
                                (layer.all_spike.sum() / layer.all_spike.view(-1).shape[0]).cpu())
                            index += 1
                    index = 1

                downsample = False  # use to downsample X
                before_X = 0  # record input before basicblock
                if (t + 1) == args.sin_t:
                    for (name_ann, layer_ann), (name_snn, layer_snn) in zip(net.named_modules(), snn.named_modules()):
                        if isinstance(layer_snn, SNode):
                            sin = float(torch.sum((layer_snn.sumspike > 0) & (x <= 0)))
                            ann = float(x[(x <= 0)].numel())
                            pos = float(torch.sum((layer_snn.sumspike > 0) & (x >= 0)))
                            layer_snn.sin_ratio.append(sin / ann)
                            layer_snn.neg_ratio.append(ann / float(layer_snn.spike.numel()))
                            layer_snn.sin_all_ratio.append(sin / float(layer_snn.spike.numel()))
                            layer_snn.pos_all_ratio.append(pos / float(layer_snn.spike.numel()))
                            layer_snn.should_all_ratio.append(
                                float(torch.sum((layer_snn.sumspike <= 0) & (x >= 0))) / float(layer_snn.spike.numel()))
                            if sin == 0.0 and pos == 0.0:
                                layer_snn.confidence.append(0.0)
                            else:
                                layer_snn.confidence.append(sin / (sin + pos))
                            if float(torch.sum(layer_snn.sumspike[(layer_snn.sumspike > 0) & (x <= 0)])) == 0.0:
                                layer_snn.avg_error_spikenum.append(0.0)
                            else:
                                layer_snn.avg_error_spikenum.append(float(torch.sum(layer_snn.sumspike[(layer_snn.sumspike > 0) & (x <= 0)])) /
                                                                sin)
                        if isinstance(layer_ann, BasicBlock):
                            before_X = x
                        if name_ann.split('.')[-1] == "downsample":
                            downsample = True
                        if not isinstance(layer_ann, nn.Sequential) and not isinstance(layer_ann,
                                                                                       VGG16) and not isinstance(
                                layer_ann, nn.ReLU) and not isinstance(layer_ann, nn.MaxPool2d) and not isinstance(layer_ann, BasicBlock)\
                                and not isinstance(layer_ann, ResNet20):
                            if isinstance(layer_ann, nn.Linear):
                                x = x.view(x.shape[0], -1)

                            if downsample is False:
                                x = layer_ann(x)
                            else:
                                before_X = layer_ann(before_X)
                            if name_ann.split('.')[-1] == "relu2":
                                x = x + before_X
                                downsample = False
        accs.append(np.array(acc))

    if True:
        f = open('{}/result.txt'.format(folder_path), 'w')
        f.write("Setting Arguments.. : {}\n".format(args))
        accs = np.array(accs).mean(axis=0)
        for iii in range(256):
            if iii == 0 or iii == 3 or iii == 7 or (iii + 1) % 16 == 0:
                f.write("timestep {}:{}\n".format(str(iii+1).zfill(3), accs[iii]))
        f.write("max accs: {}, timestep:{}\n".format(max(accs), np.where(accs == max(accs))))
        f.write("use for latex tabular: & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(max(accs) * 100, accs[31]* 100,
                                                                                         accs[63]*100, accs[127]*100, accs[255]*100))
        f.close()
        accs = torch.from_numpy(accs)
        torch.save(accs, "{}/accs.pth".format(folder_path))

        if True:
            f = open('{}/result_SINRate.txt'.format(folder_path), 'w')
            for layer_ann, layer_snn in zip(net.modules(), snn.modules()):
                if isinstance(layer_snn, SNode):
                    f.write("{:.3f}\n".format(np.mean(layer_snn.sin_ratio)))
            f.close()
            f = open('{}/result_NegRate.txt'.format(folder_path), 'w')
            for layer_ann, layer_snn in zip(net.modules(), snn.modules()):
                if isinstance(layer_snn, SNode):
                    f.write("{:.3f}\n".format(np.mean(layer_snn.neg_ratio)))
            f.close()
            f = open('{}/result_SIN_ALLRate.txt'.format(folder_path), 'w')
            for layer_ann, layer_snn in zip(net.modules(), snn.modules()):
                if isinstance(layer_snn, SNode):
                    f.write("{:.3f}\n".format(np.mean(layer_snn.sin_all_ratio)))
            f.close()
            f = open('{}/result_Pos_ALLRate.txt'.format(folder_path), 'w')
            for layer_ann, layer_snn in zip(net.modules(), snn.modules()):
                if isinstance(layer_snn, SNode):
                    f.write("{:.3f}\n".format(np.mean(layer_snn.pos_all_ratio)))
            f.close()
            f = open('{}/result_Should_ALLRate.txt'.format(folder_path), 'w')
            for layer_ann, layer_snn in zip(net.modules(), snn.modules()):
                if isinstance(layer_snn, SNode):
                    f.write("{:.3f}\n".format(np.mean(layer_snn.should_all_ratio)))
            f.close()
            f = open('{}/result_confidence.txt'.format(folder_path), 'w')
            for layer_ann, layer_snn in zip(net.modules(), snn.modules()):
                if isinstance(layer_snn, SNode):
                    f.write("{:.3f}\n".format(np.mean(layer_snn.confidence)))
            f.close()
            f = open('{}/result_avg_error_spikenum.txt'.format(folder_path), 'w')
            for layer_ann, layer_snn in zip(net.modules(), snn.modules()):
                if isinstance(layer_snn, SNode):
                    f.write("{:.3f}\n".format(np.mean(layer_snn.avg_error_spikenum)))
            f.close()
            f = open('{}/result_firingRate.txt'.format(folder_path), 'w')
            for x in range(8):
                index = 1
                f.write("-----------------timestep:{}-----------------\n".format((x + 1) * 32))
                for name, layer in snn.named_modules():
                    if isinstance(layer, SNode):
                        f.write("relu{}: average spike number: {}\n".format(index, torch.stack(
                            spike_rate_dict['relu' + str(index)][x]).mean()))
                        index += 1
            for x in range(8):
                index = 1
                f.write("-----------------timestep:{}-----------------\n".format((x + 1) * 32))
                for name, layer in snn.named_modules():
                    if isinstance(layer, SNode):
                        f.write("relu{}: average spike rate: {}\n".format(index, torch.stack(
                            spike_rate_dict['relu' + str(index)][x]).mean() / ((x + 1) * 32)))
                        index += 1

if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    seed_all(seed=args.seed)
    device = torch.device("cuda:0") if args.cuda else 'cpu'

    normalize = torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    cifar100_train = datasets.CIFAR100(root='../data/', train=True, download=False, transform=transform_train)
    cifar100_test = datasets.CIFAR100(root='../data/', train=False, download=False, transform=transform_test)
    train_iter = torch.utils.data.DataLoader(cifar100_train, batch_size=args.train_batch, shuffle=False, num_workers=0)
    test_iter = torch.utils.data.DataLoader(cifar100_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model_name == 'vgg16':
        net = VGG16()
        net.load_state_dict(torch.load("../saved_model/CIFAR100_VGG16.pth", map_location=device))
    elif args.model_name == 'resnet20':
        net = ResNet20()
        net.load_state_dict(torch.load("../saved_model/CIFAR100_ResNet20.pth", map_location=device))
    else:
        raise NameError

    net.eval()
    net = net.to(device)
    net1 = deepcopy(net)

    data = iter(test_iter).next()[0].to(device)
    data = data[0, :, :, :].unsqueeze(0)
    net(data, compute_efficiency=True)

    acc = evaluate_accuracy(test_iter, net, device)
    print("acc on ann is : {:.4f}".format(acc))

    converter = Converter(train_iter, device, args.p, args.lateral_inhi,
                          args.gamma, args.smode, args.VthHand, args.useDET, args.useDTT, args.useSC)
    snn = converter(net)  # use threshold balancing or not
    # snn = fuseConvBN(snn)

    converter = Converter(train_iter, device, args.p, args.lateral_inhi,
                          args.gamma, False, args.VthHand, args.useDET, args.useDTT, args.useSC)
    net1 = converter(net1)  # use threshold balancing or not
    evaluate_snn(test_iter, snn, net1, device, duration=args.T)

