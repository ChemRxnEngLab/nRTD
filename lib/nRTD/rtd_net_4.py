from typing import Optional
import torch.nn as nn
import torch
import numpy.typing as npt
import numpy as np


class RTDNet(nn.Module):
    """_summary_

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(
        self,
        kernel_sizes: list[int],
        padding_mode: str = "replicate",
        kernel_times: Optional[list[tuple[float, float]]] = None,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.n_compartements = len(kernel_sizes)
        self.padding_mode = padding_mode
        conv_layers = []
        for kernel_size in self.kernel_sizes:
            conv_layer = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                bias=False,
                padding=kernel_size - 1,
                padding_mode=self.padding_mode,
            )
            conv_layers.append(conv_layer)
        self.fn = nn.Sequential(*conv_layers)
        if kernel_times is None:
            self.t_conv = [(0.0, 10.0) for i in range(self.n_compartements)]
        elif len(kernel_times) != self.n_compartements:
            raise ValueError(
                "The number of compartments and the number of time intervals must be the same."
            )
        else:
            self.t_conv = kernel_times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method of the RTDNet class.

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            output
        """
        conv_out = self.fn(x)
        return conv_out

    def _freeze_conv(self, ind: int) -> None:
        """Utility function for freezing the weights of the convolutional layer.

        Parameters
        ----------
        ind : int
            index of the convolutional layer to freeze.
        """
        for i, conv in enumerate(self.fn):
            if i == ind:
                conv.weight.requires_grad = False

    @property
    def E(self) -> list[npt.NDArray]:
        """RTD density functions for compartements.

        Returns
        -------
        list[npt.NDArray]
            RTD density functions for compartements
        """
        return [
            conv.get_parameter("weight")[0, 0, :].cpu().flip(0).detach().numpy()
            for conv in self.fn
        ]

    def output_shape(self, c_in: torch.Tensor) -> int:
        """The output shape (discretiozation) of the RTD convolution function if the input shape is 'c_in'.

        Parameters
        ----------
        c : torch.Tensor
            temporal input signal

        Returns
        -------
        int
            discretization (length) of the output signal
        """
        n_c = c_in.shape[2]
        return n_c + (np.array(self.kernel_sizes) - 1).sum()
        # return n_c + self.n_compartements * (self.kernel_size + 1)

    @property
    def t_conv_end(self) -> float:
        """
        The end time of the RTD Convolution function, as the sum of the end times of the individual compartments.

        Returns
        -------
        float
            end time of the RTD Convolution function
        """
        t_conv_end = [t[-1] for t in self.t_conv]
        return sum(t_conv_end)

    def conv_t(self, t_input: torch.Tensor) -> torch.Tensor:
        """The Output time scale of the RTD convolution function if the input time scale is 't_input'.

        Parameters
        ----------
        t_input : torch.Tensor
            time scale of the input signal.

        Returns
        -------
        torch.Tensor
            time scale of the convoluted (output) signal.
        """
        t_i_end = float(t_input[-1])
        t_c_end = t_i_end + self.t_conv_end
        return torch.linspace(0, t_c_end, self.output_shape(t_input))


if __name__ == "__main__":
    fitter = RTDNet(kernel_sizes=[100], padding_mode="replicate")
    print(fitter.output_shape(torch.rand(1, 1, 100)))
    print(fitter.E)

    # forward pass
    c = fitter(torch.rand(1, 1, 100))
    print(c.shape)
