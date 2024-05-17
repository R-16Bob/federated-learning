#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        """
        训练给定网络的函数。

        参数:
        - net: 要训练的网络模型。

        返回:
        - net.state_dict(): 网络模型的状态字典，记录了训练后的参数。
        - sum(epoch_loss) / len(epoch_loss): 整个训练过程的平均损失。
        """
        net.train()  # 将网络设置为训练模式
        # 初始化优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []  # 用于记录每个epoch的损失
        for iter in range(self.args.local_ep):
            batch_loss = []  # 用于记录每个batch的损失
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # 数据加载到指定设备
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()  # 清空梯度
                log_probs = net(images)  # 通过网络计算log概率
                loss = self.loss_func(log_probs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                # 若设置为verbose模式，每隔10个batch打印一次训练状态
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())  # 记录当前batch的损失
            epoch_loss.append(sum(batch_loss)/len(batch_loss))  # 计算当前epoch的平均损失
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)  # 返回网络参数和平均损失

