import sys

sys.path.append("lib")

import torch
from nRTD import RTDModule

model = RTDModule(
    kernel_size=100,
)
