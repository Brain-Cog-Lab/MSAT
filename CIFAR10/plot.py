import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from matplotlib.pyplot import MultipleLocator
import brewer2mpl

from collections import OrderedDict
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 参照下方配色方案，第三参数为颜色数量，这个例子的范围是3-12，每种配色方案参数范围不相同
bmap = brewer2mpl.get_map('set3', 'qualitative', 12)

colors = bmap.mpl_colors


# # 一元一次函数图像
# x = np.arange(-10, 10, 0.1)
# y = 0.3 * x + np.log(1 + np.exp(x / 1.0))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(x, y)
# plt.show()
# plt.savefig("test_hx.jpg")
# sys.exit()


# have_256 = []
# root = "/home/hexiang/MSAT/CIFAR10/result_conversion_vgg16/parameters_group1/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False/"
# path = os.path.join(root, "result_avg_error_spikenum.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         have_256.append(list(map(float, numbers))[0])
# have_256 = have_256[-1]
no_sc = []
root = "/home/hexiang/MSAT/CIFAR10/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False"
path = os.path.join(root, "result_avg_error_spikenum.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        no_sc.append(list(map(float, numbers))[0])
no_sc = no_sc[-1]
have_sc = []
root = "/home/hexiang/MSAT/CIFAR10/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_True"
path = os.path.join(root, "result_avg_error_spikenum.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        have_sc.append(list(map(float, numbers))[0])
have_sc = have_sc[-1]
fig, ax = plt.subplots()
bar_width = 0.6
#
#
#
# firing_rate = [
#     0.045486677438020706, 0.07766489684581757, 0.05115384981036186,
#     0.04537772759795189, 0.04539765790104866, 0.03777753934264183,
#     0.016239548102021217, 0.026328597217798233, 0.010158049874007702,
#     0.017707584425807, 0.05968906730413437, 0.06104709580540657,
#     0.053790975362062454
# ]  # len is 13
#
# ax.bar(index, firing_rate, bar_width, color=colors[9])
# ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.set_xlabel("layer index")
# ax.set_ylabel("firing rate")
# ax.set_title('firing rate in each VGG16 layer with dt and sc')
#
# plt.show()
# plt.savefig("./firing_rate.pdf")
# sys.exit()


# ax.bar(1, have_256, bar_width, color=colors[9], label='w/o dynamic threshold and spike confidence')
ax.bar(1, no_sc, bar_width, color=colors[5], label='w dynamic threshold')
ax.bar(1 + bar_width, have_sc, bar_width, color=colors[0], label='w dynamic threshold and spike confidence')


# have_256 = []
# root = "/home/hexiang/MSAT/CIFAR10/result_conversion_resnet20/parameters_group4/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False/"
# path = os.path.join(root, "result_avg_error_spikenum.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         have_256.append(list(map(float, numbers))[0])
# have_256 = have_256[-1]
no_sc = []
root = "/home/hexiang/MSAT/CIFAR10/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False"
path = os.path.join(root, "result_avg_error_spikenum.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        no_sc.append(list(map(float, numbers))[0])
no_sc = no_sc[-1]
have_sc = []
root = "/home/hexiang/MSAT/CIFAR10/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_True"
path = os.path.join(root, "result_avg_error_spikenum.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        have_sc.append(list(map(float, numbers))[0])
have_sc = have_sc[-1]

# ax.bar(3+2*bar_width, have_256, bar_width, color=colors[9], label='w/o dynamic threshold and spike confidence')
ax.bar(3+3*bar_width, no_sc, bar_width, color=colors[5], label='w dynamic threshold')
ax.bar(3+4*bar_width, have_sc, bar_width, color=colors[0], label='w dynamic threshold and spike confidence')
ax.legend()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xticks([1 + bar_width, 3+3*bar_width], ("CIFAR10-VGG16", "CIFAR10-ResNet20")
           , fontsize=12, fontweight="normal")
plt.yticks(fontsize=12, fontweight='normal')
ax.set_ylabel("average SIN spike number")

plt.savefig("./sin_ratio_after_CIFAR10.pdf")
sys.exit()