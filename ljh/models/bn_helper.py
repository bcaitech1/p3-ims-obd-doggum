
import torch
import functools

if torch.__version__.startswith('0'):
    pass
else:
    BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    relu_inplace = True