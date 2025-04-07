# # plot for VGG-cifar100
# import torch
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import seaborn as sns
#
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1,
#                 rc={"lines.linewidth": 2.5})
#
# model_name = "resnet20"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# root = "/home/hexiang/MSAT/CIFAR100/result_conversion_{}/parameters_group1/".format(model_name)
# if model_name == "vgg16":
#     acc_target = 0.7849
# if model_name == "resnet20":
#     acc_target = 0.8069
# acc_list_target = [acc_target] * 256
#
# Path_0point5 = root + 'snn_VthHand0.2_useDET_False_useDTT_False_useSC_False/accs.pth'
# acc_list = torch.load(Path_0point5)
# acc_list1 = [acc_list[i].item() for i in range(len(acc_list))]
#
# Path_0point7 = root + 'snn_VthHand0.5_useDET_False_useDTT_False_useSC_False/accs.pth'
# acc_list = torch.load(Path_0point7)
# acc_list2 = [acc_list[i].item() for i in range(len(acc_list))]
#
# Path_0point9 = root + 'snn_VthHand0.9_useDET_False_useDTT_False_useSC_False/accs.pth'
# acc_list = torch.load(Path_0point9)
# acc_list3 = [acc_list[i].item() for i in range(len(acc_list))]
#
# Path_With_DTT = root + 'snn_VthHand-1.0_useDET_False_useDTT_True_useSC_False/accs.pth'
# acc_list = torch.load(Path_With_DTT)
# acc_list4 = [acc_list[i].item() for i in range(len(acc_list))]
#
# Path_With_DET = root + 'snn_VthHand-1.0_useDET_True_useDTT_False_useSC_False/accs.pth'
# acc_list = torch.load(Path_With_DET)
# acc_list5 = [acc_list[i].item() for i in range(len(acc_list))]
#
# Path_With_DET_DTT = root + 'snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False/accs.pth'
# acc_list = torch.load(Path_With_DET_DTT)
# acc_list6 = [acc_list[i].item() for i in range(len(acc_list))]
#
# plt.figure()
# fig, ax = plt.subplots()
# # ax.set_aspect(180)
# ax.plot(acc_list1, 'y')
# ax.plot(acc_list2, 'c')
# ax.plot(acc_list3, 'b')
# ax.plot(acc_list4, 'g')
# ax.plot(acc_list5, 'r')
# ax.plot(acc_list6, 'm')
# ax.axhline(acc_target, color='k', linestyle='--')
#
# ax1 = ax.inset_axes([0.55, 0.55, 0.3, 0.2])
# ax1.plot(acc_list1, 'y')
# ax1.plot(acc_list2, 'c')
# ax1.plot(acc_list3, 'b')
# ax1.plot(acc_list4, 'g')
# ax1.plot(acc_list5, 'r')
# ax1.plot(acc_list6, 'm')
# ax1.plot(acc_list_target, color='k', linestyle='--')
# ax1.set_xlim(224, 256)
# if model_name == "vgg16":
#     ax1.set_ylim(0.765, 0.788)
# if model_name == "resnet20":
#     ax1.set_ylim(0.795, 0.809)
# ax.indicate_inset_zoom(ax1)
#
# if model_name == "vgg16":
#     plt.legend(['0.2 x vth', '0.5 x vth', '0.9 x vth', 'With DTT',
#                 'With DET', 'With DTT DET', 'Target Acc: {}'.format(acc_target)], fontsize=10, bbox_to_anchor=[1.0, 0.43, 0, 0])
# if model_name == "resnet20":
#     plt.legend(['0.2 x vth', '0.5 x vth', '0.9 x vth', 'With DTT',
#                 'With DET', 'With DTT DET', 'Target Acc: {}'.format(acc_target)], fontsize=10, bbox_to_anchor=[1.0, 0.43, 0, 0])
# # plt.title('Spiking VGG16 on CIFAR100 Dataset')
# plt.ylim([0, 0.81])
# plt.xlim([0, 256])
# plt.xlabel('Time Step', fontsize=15, fontweight='normal')
# plt.ylabel('Top-1 Acc', fontsize=18, fontweight='normal')
# plt.xticks(fontsize=15, fontweight='normal')
# plt.yticks(fontsize=15, fontweight='normal')
# plt.subplots_adjust(bottom=0.15)
# # plt.show()
# plt.savefig('{}/acc_cifar100_{}.svg'.format(root, model_name), dpi=300)
# print('done')


# plot for SC
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import brewer2mpl
# 参照下方配色方案，第三参数为颜色数量，这个例子的范围是3-12，每种配色方案参数范围不相同
bmap = brewer2mpl.get_map('Set3', 'qualitative', 10)
colors = bmap.mpl_colors
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
datasets = "CIFAR100"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
root = "/home/hexiang/MSAT/{}/result_conversion_vgg16/parameters_group1/".format(datasets)


Path_With_DET_DTT = root + 'snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False/accs.pth'
acc_list = torch.load(Path_With_DET_DTT)
acc_list1 = [acc_list[i].item() for i in range(len(acc_list)) if (i+1) % 16 == 0 and i <= 128]

Path_With_DET_DTT_SC = root + 'snn_VthHand-1.0_useDET_True_useDTT_True_useSC_True/accs.pth'
acc_list = torch.load(Path_With_DET_DTT_SC)
acc_list2 = [acc_list[i].item() for i in range(len(acc_list)) if (i+1) % 16 == 0 and i <= 128]

root = "/home/hexiang/MSAT/{}/result_conversion_resnet20/parameters_group1/".format(datasets)
Path_With_DET_DTT = root + 'snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False/accs.pth'
acc_list = torch.load(Path_With_DET_DTT)
acc_list3 = [acc_list[i].item() for i in range(len(acc_list)) if (i+1) % 16 == 0 and i <= 128]

Path_With_DET_DTT_SC = root + 'snn_VthHand-1.0_useDET_True_useDTT_True_useSC_True/accs.pth'
acc_list = torch.load(Path_With_DET_DTT_SC)
acc_list4 = [acc_list[i].item() for i in range(len(acc_list)) if (i+1) % 16 == 0 and i <= 128]

x_list = []
for i in range(128):
    if (i + 1) % 16 == 0:
        x_list.append(i+1)
plt.figure()
fig, ax = plt.subplots()
# ax.set_aspect(180)
ax.plot(x_list[1:], acc_list1[1:], 'o-', label='w/o Spike Confidence-VGG16')
ax.plot(x_list[1:], acc_list2[1:], '^-', label='w Spike Confidence-VGG16')

ax.plot(x_list[1:], acc_list3[1:], 'o-', label='w/o Spike Confidence-ResNet20')
ax.plot(x_list[1:], acc_list4[1:], '^-', label='w Spike Confidence-ResNet20')
ax.fill_betweenx([0,1], 0, x_list[0], facecolor=colors[0], label='Spike Confidence stage')
plt.legend()
plt.grid(linestyle="--", linewidth=0.5)  # 设置背景网格线为虚线
# plt.title('Spiking VGG16 on CIFAR100 Dataset')
plt.ylim([0.55, 0.85])
plt.xlim([0, 144])
plt.xlabel('Inference timesteps', fontsize=18, fontweight='normal')
plt.ylabel('Top-1 Acc', fontsize=18, fontweight='normal')
plt.xticks(x_list, ("16", "32", "48", "64", "80", "96", "112", "128"), fontsize=12, fontweight='normal')
plt.yticks(fontsize=12, fontweight='normal')
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=15, fontweight='normal')  # 设置图例字体的大小和粗细
plt.savefig('./acc_cifar100_SC.svg', dpi=300)
print('done')