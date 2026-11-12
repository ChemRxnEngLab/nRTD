"""Version 2 of rtd_net: the kernel can be fitted in logit space (softmax), so E ≥ 0 and Σ E·Δt = 1 hold exactly.

Differences to rtd_net:
  - `RTDConv(..., positivity="softmax")` (default) parametrizes the convolution weight as
        weight = softmax(θ) / Δt,
    θ being the trainable logits (`conv.parametrizations.weight.original`). The weight is the flipped E; the softmax
    does not depend on the order, so E = softmax(θ_flipped) / Δt as well. Setting `E` (or `set_to_direct_response`)
    stores θ = log(max(E·Δt, floor)), i.e. E is normalized to Σ E·Δt = 1 and values below `floor / Δt` are raised to it.
  - `positivity="clamp"` keeps the free weights of rtd_net (non-negativity then is up to the fitting module).
  - `RTDNet` passes `positivity` and `floor` to all its layers and uses its default kernel times when none are given
    (rtd_net zips over the `kernel_times` argument, which fails for `None`).
"""

import logging
from typing import Optional
import torch.nn as nn
import torch
import torch.nn.utils.parametrize as parametrize


POSITIVITY = ("softmax", "clamp")


def vec_linspace(
    starts: torch.Tensor,
    stops: torch.Tensor,
    num: int,
    endpoint: bool = True,
):
    if endpoint:
        devisor = num - 1
    else:
        devisor = num
    steps = (1.0 / devisor) * (stops - starts)
    return steps[:, None] * torch.arange(num, device=starts.device) + starts[:, None]


class TimescaleError(Exception):
    pass


class SoftmaxKernel(nn.Module):
    """Parametrization weight = softmax(θ) / Δt over the kernel axis: positive, Σ weight·Δt = 1.

    Parameters
    ----------
    delta_t : float
        time step of the kernel
    floor : float
        smallest E·Δt a kernel point can be set to (log 0 is not representable)
    """

    def __init__(self, delta_t: float, floor: float = 1e-12):
        super().__init__()
        self.delta_t = float(delta_t)
        self.floor = float(floor)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.softmax(theta, dim=-1) / self.delta_t

    def right_inverse(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(weight * self.delta_t, min=self.floor))


class RTDConv(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        padding_mode: str = "replicate",
        kernel_time: tuple[float, float] = (0.0, 10.0),
        positivity: str = "softmax",
        floor: float = 1e-12,
    ):
        super().__init__()
        if positivity not in POSITIVITY:
            raise ValueError(f"positivity must be one of {POSITIVITY}.")
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.kernel_time = kernel_time
        self.positivity = positivity
        self.delta_t = torch.tensor(
            (kernel_time[1] - kernel_time[0]) / (kernel_size - 1)
        )
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=False,
            padding=kernel_size - 1,
            padding_mode=self.padding_mode,
        )
        if self.positivity == "softmax":
            parametrize.register_parametrization(
                self.conv, "weight", SoftmaxKernel(self.delta_t.item(), floor)
            )
        self._frozen = False

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # calculate the difference in the time arrays along the 1st dimension (shape is n_samples x n_in)
        delta_t_data = t.diff(dim=1)[:, 0]

        if x.shape[2] != t.shape[1]:
            raise TimescaleError(
                "The length of the input signal and the input time scale must be the same."
            )
        if not torch.isclose(delta_t_data, self.delta_t).all():
            raise TimescaleError(
                "The time scale of the input signal and the convolution kernel must be same."
            )
        conv_out = self.conv(x) * self.delta_t

        return conv_out, self.conv_t(t)

    def output_shape(self, input_or_shape: torch.Tensor | tuple[int, ...] | int) -> int:
        """The output discretization of the RTD convolution function if the input shape is 'c_in'.

        Parameters
        ----------
        c : torch.Tensor
            temporal input signal

        Returns
        -------
        int
            discretization (length) of the output signal
        """
        if isinstance(input_or_shape, tuple):
            n_c = input_or_shape[2]
        elif isinstance(input_or_shape, int):
            n_c = input_or_shape
        elif isinstance(input_or_shape, torch.Tensor):
            n_c = input_or_shape.shape[2]
        else:
            raise ValueError("The input must be a torch.Tensor or a torch.Size object.")
        return n_c + self.kernel_size - 1

    @property
    def E(self) -> torch.Tensor:
        """RTD density function of the compartement.

        Returns
        -------
        torch.Tensor
            RTD density function of the compartement
        """
        return self.conv.weight.detach()[0, 0, :].cpu().flip(0)

    @E.setter
    def E(self, E_new: torch.Tensor) -> None:
        # flip the kernel and add two dimensions
        weight = E_new.flip(0).unsqueeze(0).unsqueeze(0).to(self.delta_t.dtype)
        with torch.no_grad():
            if self.positivity == "softmax":
                # stores θ via SoftmaxKernel.right_inverse
                self.conv.weight = weight.to(self.conv.parametrizations.weight.original.device)
            else:
                self.conv.weight.data = weight.to(self.conv.weight.device)

    def set_to_direct_response(self):
        E = torch.zeros(self.kernel_size)
        E[0] = 1.0 / self.delta_t
        self.E = E

    @property
    def t_kernel(self) -> torch.Tensor:
        """The time scale of the convolution kernel.

        Returns
        -------
        torch.Tensor
            time scale of the convolution kernel
        """
        return torch.linspace(
            self.kernel_time[0], self.kernel_time[1], self.kernel_size
        )

    def conv_t(self, t_input: torch.Tensor) -> torch.Tensor:
        """The Output time scale of the RTD convolution function if the input time scale is 't_input'.

        Parameters
        ----------
        t_input : torch.Tensor
            time scale of the input signal.

        Returns
        -------
        torch.Tensor
            time scale of the convolved (output) signal.
        """
        t_c_start = t_input[:, 0] + self.kernel_time[0]
        t_c_end = t_input[:, -1] + self.kernel_time[1]

        return vec_linspace(t_c_start, t_c_end, self.output_shape(t_input.shape[1]))

    def clamp_(self) -> None:
        """Set negative kernel values to 0 (positivity 'clamp' only; softmax kernels are positive)."""
        if self.positivity == "clamp":
            self.conv.weight.data.clamp_(min=0)

    def freeze(self) -> None:
        for name, p in self.named_parameters():
            logging.debug(f"Froze parameter {name}.")
            p.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        for name, p in self.named_parameters():
            logging.debug(f"Unfroze parameter {name}.")
            p.requires_grad = True
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    @frozen.setter
    def frozen(self, freeze: bool) -> None:
        if freeze:
            self.freeze()
        else:
            self.unfreeze()


class RTDNet(nn.Module):
    """Compartments in series, one RTDConv each.

    Parameters
    ----------
    kernel_sizes : list[int]
        number of kernel points per compartment
    kernel_times : list[tuple[float, float]], optional
        time window per compartment, (0, 10) each if not given
    positivity : str
        'softmax' (E in logit space) or 'clamp' (free weights), for all compartments
    floor : float
        smallest E·Δt a softmax kernel point can be set to
    """

    def __init__(
        self,
        kernel_sizes: list[int],
        kernel_times: Optional[list[tuple[float, float]]] = None,
        positivity: str = "softmax",
        floor: float = 1e-12,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.n_compartements = len(kernel_sizes)
        self.positivity = positivity
        self.kernel_times: list[tuple[float, float]]
        if kernel_times is None:
            self.kernel_times = [(0.0, 10.0) for i in range(self.n_compartements)]
        elif len(kernel_times) != self.n_compartements:
            raise ValueError(
                "The number of compartments and the number of time intervals must be the same."
            )
        else:
            self.kernel_times = kernel_times

        self.conv_layers = nn.ModuleList()
        for kernel_size, kernel_time in zip(self.kernel_sizes, self.kernel_times):
            self.conv_layers.append(
                RTDConv(
                    kernel_size=kernel_size,
                    padding_mode="replicate",
                    kernel_time=kernel_time,
                    positivity=positivity,
                    floor=floor,
                )
            )

        self.delta_t = [conv.delta_t for conv in self.conv_layers]

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward method of the RTDNet class.

        Parameters
        ----------
        x : torch.Tensor
            input data

        t: torch.Tensor
            time scale of the input data

        Returns
        -------
        torch.Tensor
            output

        torch.Tensor
            time scale of the output
        """
        for conv in self.conv_layers:
            x, t = conv(x, t)
        return x, t

    def partial_forward(self, x, t, begin=None, end=None):
        if begin is None:
            begin = 0
        if end is None:
            end = len(self.conv_layers)

        for conv in self.conv_layers[begin:end]:
            x, t = conv(x, t)
        return x, t

    def _freeze_conv(self, ind: int) -> None:
        """Utility function for freezing the weights of the convolutional layer.

        Parameters
        ----------
        ind : int
            index of the convolutional layer to freeze.
        """
        self.conv_layers[ind].freeze()

    def clamp_(self) -> None:
        for conv in self.conv_layers:
            conv.clamp_()

    @property
    def E(self) -> list[torch.Tensor]:
        """RTD density functions for compartements.

        Returns
        -------
        list[torch.Tensor]
            RTD density functions for compartements
        """
        return [conv.E for conv in self.conv_layers]

    def output_shape(self, input_or_shape: torch.Tensor | tuple[int, ...] | int) -> int:
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
        if isinstance(input_or_shape, tuple):
            n_c = input_or_shape[2]
        elif isinstance(input_or_shape, int):
            n_c = input_or_shape
        elif isinstance(input_or_shape, torch.Tensor):
            n_c = input_or_shape.shape[2]
        else:
            raise ValueError("The input must be a torch.Tensor or a torch.Size object.")

        for conv in self.conv_layers:
            n_c = conv.output_shape(n_c)
        return n_c

    @property
    def t_conv_end(self) -> float:
        """
        The end time of the RTD Convolution function, as the sum of the end times of the individual compartments.

        Returns
        -------
        float
            end time of the RTD Convolution function
        """
        t_conv_end = [t[-1] for t in self.kernel_times]
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
        t = t_input
        for conv in self.conv_layers:
            t = conv.conv_t(t)
        return t


if __name__ == "__main__":
    pass
