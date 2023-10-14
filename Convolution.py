import torch.nn as nn

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Convolution, self).__init__()
        self.E = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        conv_E = nn.functional.conv2d(x, self.E)
        conv_out = self.conv(x)
        return conv_E + conv_out
