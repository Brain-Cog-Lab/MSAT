import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from matplotlib.pyplot import MultipleLocator


# no_16 = []
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group3/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False"
# path = os.path.join(root, "result_SINRate.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         no_16.append(list(map(float, numbers))[0])
#
# have_16 = []
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False"
# path = os.path.join(root, "result_SINRate.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         have_16.append(list(map(float, numbers))[0])
#
# have_32 = []
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group2/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False"
# path = os.path.join(root, "result_SINRate.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         have_32.append(list(map(float, numbers))[0])
index = np.arange(1, 14)
#
# fig, ax = plt.subplots()
# bar_width = 0.3
#
# ax.bar(index - bar_width, have_16, bar_width, color='m', label='timestep 16')
# ax.bar(index, have_32, bar_width, color='r', label='timestep 32')
# ax.bar(index + bar_width, no_16, bar_width, color='b', label='timestep 256')
# ax.legend()
#
# ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.set_xlabel("layer index", fontsize=18, fontweight='normal')
# ax.set_ylabel("SIN ratio", fontsize=18, fontweight='normal')
# # ax.set_title('sin ratio in each ResNet20 layer')
# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=14, fontweight='normal')  # 设置图例字体的大小和粗细
# plt.show()
# plt.savefig("./sin_ratio_vgg16.svg", dpi=800)
#
# no_16 = []
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_resnet20/parameters_group1/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False"
# path = os.path.join(root, "result_SINRate.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         no_16.append(list(map(float, numbers))[0])
#
# have_16 = []
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_resnet20/parameters_group3/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False"
# path = os.path.join(root, "result_SINRate.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         have_16.append(list(map(float, numbers))[0])
#
# have_32 = []
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_resnet20/parameters_group4/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False"
# path = os.path.join(root, "result_SINRate.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         have_32.append(list(map(float, numbers))[0])
# index = np.arange(1, 20)
#
# fig, ax = plt.subplots()
# bar_width = 0.3
#
#
# ax.bar(index - bar_width, have_16, bar_width, color='m', label='timestep 32')
# ax.bar(index, have_32, bar_width, color='r', label='timestep 48')
# ax.bar(index + bar_width, no_16, bar_width, color='b', label='timestep 256')
# ax.legend()
#
# ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.set_xlabel("layer index", fontsize=18, fontweight='normal')
# ax.set_ylabel("SIN ratio", fontsize=18, fontweight='normal')
# # ax.set_title('sin ratio in each ResNet20 layer')
# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=14, fontweight='normal')  # 设置图例字体的大小和粗细
#
# plt.show()
# plt.savefig("./sin_ratio_resnet20.svg", dpi=800)
# sys.exit()
# # #
# #
# # Vth = []
# # Vmem = []
# # Vrd = []
# # plt.figure()
# # fig, ax = plt.subplots()
# #
# # with open('Vmem_timestep.txt', 'r') as f:
# #     data = f.readlines()  # 将txt中所有字符串读入data
# #
# #     for ind, line in enumerate(data):
# #         numbers = line.split()  # 将数据分隔
# #         Vmem.append(list(map(float, numbers))[1])  # 转化为浮点数
# #         if ind == 255:
# #             ax.plot(Vmem, 'g')
# #             break
# #
# # with open('Vrd_timestep.txt', 'r') as f:
# #     data = f.readlines()  # 将txt中所有字符串读入data
# #
# #     for ind, line in enumerate(data):
# #         numbers = line.split()  # 将数据分隔
# #         Vrd.append(list(map(float, numbers))[1])  # 转化为浮点数
# #         if ind == 255:
# #             ax.plot(Vrd, 'k')
# #             break
# #
# # with open('Vth_timestep.txt', 'r') as f:
# #     data = f.readlines()  # 将txt中所有字符串读入data
# #
# #     for ind, line in enumerate(data):
# #         numbers = line.split()  # 将数据分隔
# #         if ind > 0:
# #             Vth.append(list(map(float, numbers))[1])  # 转化为浮点数
# #         if ind == 255:
# #             ax.plot(Vth, 'r')
# #             break
# #
# #
# # plt.legend(['Vmem', 'Vrd', 'Vth'], fontsize=10)
# # # plt.title('Spiking VGG16 on CIFAR100 Dataset')
# # plt.xlabel('Time Step', fontsize=11)
# # plt.ylabel('Value', fontsize=11)
# #
# # plt.savefig('./vth.pdf', dpi=600)
# # print('done')
#
#
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
#
#
# #
# #
# # have_256 = []
# # root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False/"
# # path = os.path.join(root, "result_avg_error_spikenum.txt")
# # with open(path, 'r') as f:
# #     data = f.readlines()  # 将txt中所有字符串读入data
# #     for ind, line in enumerate(data):
# #         numbers = line.split()  # 将数据分隔
# #         have_256.append(list(map(float, numbers))[0])
# # have_256 = have_256[-1]
# no_sc = []
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False"
# path = os.path.join(root, "result_avg_error_spikenum.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         no_sc.append(list(map(float, numbers))[0])
# no_sc = no_sc[-1]
# have_sc = []
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_True"
# path = os.path.join(root, "result_avg_error_spikenum.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         have_sc.append(list(map(float, numbers))[0])
# have_sc = have_sc[-1]
fig, ax = plt.subplots()
bar_width = 0.6
# #
# #
# #
# # firing_rate = [
# #     0.045486677438020706, 0.07766489684581757, 0.05115384981036186,
# #     0.04537772759795189, 0.04539765790104866, 0.03777753934264183,
# #     0.016239548102021217, 0.026328597217798233, 0.010158049874007702,
# #     0.017707584425807, 0.05968906730413437, 0.06104709580540657,
# #     0.053790975362062454
# # ]  # len is 13
# #
firing_rate = [
    0.09701524674892426, 0.09603047370910645, 0.06812159717082977,
    0.08169744163751602, 0.08108360320329666, 0.05183771252632141,
    0.050302114337682724, 0.024889685213565826, 0.017198534682393074,
    0.010185545310378075, 0.02891930378973484, 0.026949070394039154,
    0.12731415033340454
] # len is 13
ax.bar(index, firing_rate, bar_width, color=colors[9])
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_xlabel("Layer index", fontsize=14)
ax.set_ylabel("Firing rate", fontsize=14)
# ax.set_title('firing rate in each layer with our methods')

plt.show()
plt.savefig("./firing_rate.pdf")
sys.exit()
#
#
# # ax.bar(1, have_256, bar_width, color=colors[9], label='w/o dynamic threshold and spike confidence')
# ax.bar(1, no_sc, bar_width, color=colors[5], label='w dynamic threshold')
# ax.bar(1 + bar_width, have_sc, bar_width, color=colors[0], label='w dynamic threshold and spike confidence')
#
#
# have_256 = []
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_resnet20/parameters_group1/snn_VthHand1.0_useDET_False_useDTT_False_useSC_False/"
# path = os.path.join(root, "result_avg_error_spikenum.txt")
# with open(path, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for ind, line in enumerate(data):
#         numbers = line.split()  # 将数据分隔
#         have_256.append(list(map(float, numbers))[0])
# have_256 = have_256[-1]
no_sc = []
root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False"
path = os.path.join(root, "result_avg_error_spikenum.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        no_sc.append(list(map(float, numbers))[0])
no_sc = no_sc[-1]
have_sc = []
root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_True"
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

plt.xticks([1 + bar_width, 3+3*bar_width], ("CIFAR100-VGG16", "CIFAR100-ResNet20")
           , fontsize=12, fontweight="normal")
plt.yticks(fontsize=12, fontweight='normal')
ax.set_ylabel("average SIN spike number")

plt.savefig("./sin_ratio_after_CIFAR100.pdf")
sys.exit()

Which = 256  # or 256; 256 is abnormal
Vmem = []
root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False"
path = os.path.join(root, "Vmem_timestep.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        if ind >= Which:
            Vmem.append(list(map(float, numbers))[3])
        if ind == Which + 127:
            break

minCoor = np.min(np.array(Vmem))
print(minCoor)
Vth = []
root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False"
path = os.path.join(root, "Vth_timestep.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        if ind >= Which:
            Vth.append(list(map(float, numbers))[3])
        if ind == Which + 127:
            break

Spike = []
root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False"
path = os.path.join(root, "Spike_timestep.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        if ind >= Which:
            spike = list(map(float, numbers))[3]
            if spike != 0.0:
                plt.axvline(ind - Which, label="spike", ymin=0, ymax=0.05, c='r')
        if ind == Which + 127:
            break



x_list = np.arange(0, 128)
plt.plot(x_list, Vmem, c=colors[3], label="Vmem")
plt.plot(x_list, Vth, c='b', label="Vth", linestyle="--")
plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('Inference timesteps', fontsize=22, fontweight='normal')
plt.ylabel('Voltage Potential', fontsize=22, fontweight='normal')
plt.xticks([0, 16, 32, 48, 64, 80, 96, 112, 128], ("0", "16", "32", "48", "64", "80", "96", "112", "128"), fontsize=14, fontweight='normal')
plt.yticks(fontsize=14, fontweight='normal')
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=14, fontweight='normal')  # 设置图例字体的大小和粗细
plt.subplots_adjust(bottom=0.15)
plt.savefig('./SIN_Show_AbNormal.svg', dpi=300)