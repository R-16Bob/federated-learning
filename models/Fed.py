#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])  # 也就是第一个客户端的模型参数（状态字典）
    for k in w_avg.keys():  # 遍历状态字典的键
        for i in range(1, len(w)):  # 遍历剩下的客户端
            w_avg[k] += w[i][k]  # 模型参数相加
        w_avg[k] = torch.div(w_avg[k], len(w))  # 模型参数相加后除以客户端个数，做FedAvg
    return w_avg
