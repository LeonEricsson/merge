import torch.nn.functional as F

ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": F.tanh,
    "silu": F.silu,
    "sigmoid": F.sigmoid,
    "identity": lambda x: x,
    "none": lambda x: x,
}
