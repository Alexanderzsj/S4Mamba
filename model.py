import torch
import torch.nn as nn
from calflops import calculate_flops


"""
Description
args.model_type=4: used for mamba net, whole image is used as input


args.model_type=1: For spatial data，只适合模型的输出为[B, num_classes]的情况
args.model_type=1 and model_3D_spa=1: For 3D spatial data
args.model_type=2: For spectral data
args.model_type=1: For spatial and spectral data
"""


def build_model(args):
    """
    Build the model based on the specified args.
    Args:
        args: the arguments parsed from command line.
    Returns: the model.
    """
    model = None

    if args.model.startswith('S4Mamba'):
        from models.S4Mamba.S4Mamba import S4Mamba
        args.model_type=1 # load spatial data, function: generate_spatial_iter
        args.model_3D_spa=0 # laod 3D data
        args.step_size=50
        args.gamma=0.99
        model = S4Mamba(in_channels=args.channels, num_classes=args.num_classes, hidden_dim=128)



    return model
    