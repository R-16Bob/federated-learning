#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    """
    测试模型的准确率和损失。

    参数:
    - net_g: 训练好的生成网络，用于测试。
    - datatest: 测试数据集，包含图片和对应的标签。
    - args: 包含各种设置的参数对象，如批处理大小(bs)，gpu使用情况等。

    返回:
    - accuracy: 测试集上的准确率。
    - test_loss: 测试集上的平均损失。
    """
    net_g.eval()  # 将网络设置为评估模式
    # 初始化测试损失和正确数目
    test_loss = 0
    correct = 0
    # 使用DataLoader加载测试数据
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)  # 测试数据集的批次数量

    # 遍历测试数据集
    for idx, (data, target) in enumerate(data_loader):
        # 如果使用gpu，则将数据和目标移动到gpu
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        # 通过网络获取log概率
        log_probs = net_g(data)
        # 计算批次损失并累加
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item() # 将张量转换为python基础类型
        # 预测类别并计算正确数目
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    # 计算平均损失和准确率
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    # 如果设置为verbose，打印测试结果
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


