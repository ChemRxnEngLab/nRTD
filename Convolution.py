import torch.nn as nn

class RTD_Fitter(nn.Module):
    def __init__(self, kernel_size, padding_mode = "replicate", n_compartements = 1):
        super(RTD_Fitter, self).__init__()
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.fn = nn.Sequential(
            *[nn.Conv1D(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                bias=False, # no offset
                padding=self.kernel_size, # we append n_kernel values to the left and right of the input to make sure the convolution is causal/full
                padding_mode=self.padding_mode
            ) for i in range(n_compartements)]
        )
        # self.conv = nn.Conv1D(
        #     in_channels=1,
        #     out_channels=1,
        #     kernel_size=kernel_size,
        #     bias=False, # no offset
        #     padding=self.kernel_size, # we append n_kernel values to the left and right of the input to make sure the convolution is causal/full
        #     padding_mode=self.padding_mode
        # )

    def forward(self, x):
        conv_out = self.fn(x)
        return conv_out

    @property
    def E(self):
        return self.conv.weight[0,0,:].flip(0).detach().numpy()