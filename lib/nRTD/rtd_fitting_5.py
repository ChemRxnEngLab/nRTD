#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:50:14 2024

@author: tuanaoyuncu
"""

from typing import Any, Optional
import torch
import lightning.pytorch as pl
from . import rtd_net_5 as rtd_net
from pathlib import Path
import numpy as np
import scipy


## this is only for the Capillary data!
class RTDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_file: Path,
        switching_times: list[float],
        t_range_in: tuple[float, float],
        n_in: int,
        t_range_out: tuple[float, float],
        n_out: int,
        switch_delay: float = 0.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_file = data_file
        # calculate the absolute times from the relative times between two switching events
        self.switching_times = switching_times
        self.intervals = [
            (self.switching_times[i], self.switching_times[i + 1])
            for i in range(len(self.switching_times) - 1)
        ]
        self.t_range_in = t_range_in
        self.t_range_out = t_range_out

        self.n_out = n_out
        self.n_in = n_in

        # self.n_in = int((self.t_range_in[1] - self.t_range_in[0]) * n_out)
        # self.n_out = int((self.t_range_out[1] - self.t_range_out[0]) * n_out)
        self.switch_delay = (
            switch_delay  # delay between the button click and the actual pshhht
        )
        self.is_setup = False

    def setup(self, stage: str):
        if self.is_setup:
            return False
        # load the raw data from disc
        with np.load(self.data_file) as data:
            n_dot = data["n_dot"]
            times = data["time_delta"]

        # interpolate the data to the measured time scale
        interp = scipy.interpolate.interp1d(
            times,
            n_dot,
            kind="linear",
            fill_value="extrapolate",
        )
        # calculate the time scales for the input and output data
        t_in = np.linspace(
            self.t_range_in[0],
            self.t_range_in[1],
            self.n_in,
        )
        t_out = np.linspace(
            self.t_range_out[0],
            self.t_range_out[1],
            self.n_out,
        )

        # for each interval in ther switching periods, interpolate the data on that interval
        for i, interval in enumerate(self.intervals):
            # check if the period is even or odd
            even_period = i % 2 == 0
            # begin of interpolation
            interp_begin = interval[0] + self.t_range_out[0]
            # end of interpolation
            interp_end = interp_begin + self.t_range_out[-1]
            t_interp = np.linspace(
                interp_begin,
                interp_end,
                self.n_out,
            )
            # we only care about Ar and He as tracers
            n_dot_interp = interp(t_interp)[
                [0, 1]
            ]  # shape is (2, n_out) that is two valid samples

            # get the respective 0 and 1 values for the switch
            if even_period:
                try:
                    even_steady_state_values = np.append(
                        even_steady_state_values, n_dot_interp[None, :, [0, -1]], axis=0
                    )
                except NameError:
                    even_steady_state_values = n_dot_interp[None, :, [0, -1]]
            else:
                try:
                    odd_steady_state_values = np.append(
                        odd_steady_state_values, n_dot_interp[None, :, [0, -1]], axis=0
                    )
                except NameError:
                    odd_steady_state_values = n_dot_interp[None, :, [0, -1]]

            # stack the interpolated data
            try:
                y = np.vstack((y, n_dot_interp))
            except NameError:
                y = n_dot_interp

        #  take the mean over all samples to get somewhat reliable steady state values
        even_steady_state_values = even_steady_state_values.mean(
            axis=0
        )  # shape (2,2) first index is species, second index is the steady state
        odd_steady_state_values = odd_steady_state_values.mean(axis=0)

        # claculate the step functions for each interval
        for i, interval in enumerate(self.intervals):
            # check if the period is even or odd
            even_period = i % 2 == 0

            if even_period:
                steady_state_values = even_steady_state_values
            else:
                steady_state_values = odd_steady_state_values

            n_dot_in = np.zeros((2, self.n_in))
            n_dot_in[0, :] = np.where(
                t_in < self.switch_delay,
                steady_state_values[0, 0],
                steady_state_values[0, 1],
            )
            n_dot_in[1, :] = np.where(
                t_in < self.switch_delay,
                steady_state_values[1, 0],
                steady_state_values[1, 1],
            )

            # stack the step functions
            try:
                x = np.vstack((x, n_dot_in))
            except NameError:
                x = n_dot_in
        self.y = torch.Tensor(y)[:, None, :]
        self.x = torch.tensor(x)[:, None, :]
        t_in = torch.tensor(t_in)
        t_out = torch.tensor(t_out)
        self.t_in = torch.tile(t_in, (2 * len(self.intervals), 1))
        self.t_out = torch.tile(t_out, (2 * len(self.intervals), 1))
        self.initialize_data_sets()
        self.is_setup = True
        return True

    def initialize_data_sets(self):
        self.data_set = torch.utils.data.TensorDataset(self.x, self.t_in, self.y)
        self.train_set, self.test_set = torch.utils.data.random_split(
            self.data_set,
            [
                int(0.8 * len(self.data_set)),
                len(self.data_set) - int(0.8 * len(self.data_set)),
            ],
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False
        )


# this is for the rest of results
class otherRTDDataModule(RTDDataModule):

    def __init__(
        self,
        batch_size: int,
        data_file: Path,
        switching_periods: list[float],
        t_range_in: tuple[float, float],
        n_in: int,
        t_range_out: tuple[float, float],
        n_out: int,
        switch_delay: float = 0.0,
    ):
        pass


class RTDModule(pl.LightningModule):
    def __init__(
        self,
        kernel_sizes: list[int],
        kernel_times: Optional[list[tuple[float, float]]] = None,
        learning_rate: float = 1e-3,
        use_scheduler: bool = False,
        scheduler_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.kernel_sizes = kernel_sizes
        self.kernel_times = kernel_times
        self.net = rtd_net.RTDNet(
            kernel_sizes=self.kernel_sizes,
            kernel_times=self.kernel_times,
        )

        if use_scheduler and scheduler_kwargs is None:
            raise ValueError(
                "scheduler_kwargs must be provided when use_scheduler is True."
            )

        self.use_scheduler = use_scheduler
        self.scheduler_kwargs: dict[str, Any] = scheduler_kwargs  # type: ignore

        self.save_hyperparameters()

    def forward(
        self,
        X: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(X, t)

    def training_step(self, batch, batch_idx):
        X, t, y = batch
        ### just a test for clamping the E
        for layer in self.net.conv_layers:
            layer.conv.weight.data.clamp_(min=0)
        y_hat, _ = self(X, t)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    # def training_step(self, batch, batch_idx):
    #     X, y = batch
    #     X_arranged = X[:, :, :50]
    #     y_arranged = y[:, :, :50]
    #     y_hat_arranged = self(X_arranged)
    #     y_hat_arranged = y_hat_arranged[:, :, :50]
    #     loss = torch.nn.functional.mse_loss(y_hat_arranged, y_arranged)
    #     self.log("train/loss", loss)
    #     return loss

    def validation_step(self, batch, batch_idx):
        X, t, y = batch
        y_hat, _ = self(X, t)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val/loss", loss)
        # return loss

    def test_step(self, batch, batch_idx):
        X, t, y = batch
        y_hat, _ = self(X, t)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test/loss", loss)
        # return loss

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore
        _optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if not self.use_scheduler:
            ret_dict = {"optimizer": _optimizer}
        else:
            _scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                _optimizer,
                **self.scheduler_kwargs,
            )
            ret_dict = {
                "optimizer": _optimizer,
                "lr_scheduler": _scheduler,
                "monitor": "train/loss",
            }
        return ret_dict
