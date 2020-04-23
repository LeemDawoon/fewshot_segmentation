import os
import random
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_device(gpus):
    print('gpus', gpus)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # "0, 5, 6"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(torch.cuda.device_count())
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

