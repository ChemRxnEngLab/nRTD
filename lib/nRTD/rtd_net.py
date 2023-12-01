import torch.nn as nn
import torch
import numpy.typing as npt
from typing import Optional


class RTDNet(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        padding_mode: str = "replicate",
        n_compartements: int = 1,
        t_conv: Optional[list[tuple[float, float]]] = None,
    ):
        super(RTDNet, self).__init__()
        self.n_compartements = n_compartements
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.fn = nn.Sequential(
            *[
                nn.Conv1d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=kernel_size,
                    bias=False,  # no offset
                    padding=self.kernel_size,  # we append n_kernel values to the left and right of the input to make sure the convolution is causal/full
                    padding_mode=self.padding_mode,
                )
                for i in range(n_compartements)
            ]
        )
        if t_conv is None:
            self.t_conv = [(0, 10) for i in range(n_compartements)]
        else:
            self.t_conv = t_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.fn(x)
        return conv_out

    def freeze_conv(self, ind: int) -> None:
        for i, conv in enumerate(self.fn):
            if i == ind:
                conv.weight.requires_grad = False

    @property
    def E(self) -> list[npt.NDArray]:
        return [
            conv.get_parameter("weight")[0, 0, :].flip(0).detach().numpy()
            for conv in self.fn
        ]

    def output_shape(self, c: torch.Tensor) -> int:
        """
        Returns the output shape of the convolutional layer
        """
        n_c = c.shape[2]
        return n_c + self.n_compartements * (self.kernel_size + 1)

    def conv_t(self, t_input: torch.Tensor) -> torch.Tensor:
        """
        Returns the timescale of the output signal if the input is on timescale `t_input`
        """
        t_i_end = float(t_input[-1])
        t_conv_end = [t[-1] for t in self.t_conv]
        t_c_end = t_i_end + sum(t_conv_end)
        return torch.linspace(0, t_c_end, self.output_shape(t_input))


if __name__ == "__main__":
    import torch

    fitter = RTDNet(kernel_size=100, padding_mode="replicate", n_compartements=1)
    print(fitter.E)

    # forward pass
    c = fitter(torch.rand(1, 1, 100))
    print(c.shape)
