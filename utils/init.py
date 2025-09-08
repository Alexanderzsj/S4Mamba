import torch
import torch.nn as nn
import torch.nn.init as init


def init_weights(m):
    """
    对 Conv2d 和 Linear 层的权重应用 Kaiming Normal 初始化，
    并对 GroupNorm 层的权重和偏置进行初始化。
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)