import sys
from types import MethodType

sys.path.append(r"D:\Users\Hannes\Documents\Python Code\nRTD\lib")
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.utils
import torch.utils.data
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

torch.set_default_dtype(torch.float64)

from nRTD.rtd_fitting_5 import RTDModule, RTDDataModule
from SweepRunner import Sweeper
import matplotlib.pyplot as plt

plt.style.use("ICIWstyle")

### Constants
# givens
t_out = (0.0, 120.0)
n_out = int(4 * (t_out[1] - t_out[0]) + 1)
delta_t_out = (t_out[1] - t_out[0]) / (n_out - 1)

# the capillary times
switching_periods = np.array(
    [
        45.35,
        45.18,
        45.06,
        44.98,
        45.20,
        45.16,
        45.11,
        45.03,
        45.10,
        45.15,
    ]
)
switching_times_cap = np.flip(np.cumsum(-switching_periods))
switching_times_cap = np.append(switching_times_cap, 0)

# the piping times
t_offset = 6.5
t_halfperiod = t_out[1] - t_out[0]

switching_times_reac_inlet = np.full((20,), t_halfperiod)
switching_times_reac_inlet = np.insert(switching_times_reac_inlet, 0, t_offset)
switching_times_reac_inlet = np.cumsum(switching_times_reac_inlet)


t_kernels = [
    (0.0, 25.0),  # piping to reactor inlet
    (0.0, 25.0),  # capillary
]

n_kernels = list(
    map(
        lambda t_kernel: int(((t_kernel[1] - t_kernel[0]) / delta_t_out) + 1), t_kernels
    )
)

t_in = (0, t_out[1] - sum([t_kernel[1] for t_kernel in t_kernels]))
n_in = int(((t_in[1] - t_in[0]) / delta_t_out) + 1)

wandb.init(project="HSA_MGA_nRTD", entity="ice_ulm")

data_cap = RTDDataModule(
    batch_size=32,
    data_file=Path(
        r"D:\Users\Hannes\Documents\Python Code\nRTD\Experiments\HSA_001\data\MGA-E-05032025_300ml-min_Analytik_MS.npz"
    ),
    switching_times=switching_times_cap,
    t_range_in=t_in,
    n_in=n_in,
    t_range_out=t_out,
    n_out=n_out,
    switch_delay=0.7,
)


def fake_setup(self, stage: str):
    import scipy

    print("Setup")
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
    delta_t = (self.t_range_out[1] - self.t_range_out[0]) / (self.n_out - 1)

    # for each interval in ther switching periods, interpolate the data on that interval
    for i, interval in enumerate(self.intervals):
        # check if the period is even or odd
        even_period = i % 2 == 0
        # begin of interpolation
        interp_begin = interval[0] + self.t_range_out[0]
        # end of interpolation
        interp_end = interp_begin + self.t_range_out[-1]
        # auxilliray time scale for the interpolation
        # 15 seconds of the end of each signal will be repeated to pad out the signal to 2 min
        interp_aux = interp_begin + 30
        # available time scale for interpolation
        #  hard coded 15 seconds
        interp_avail = interp_begin + 45

        t_interp = np.linspace(
            interp_begin,
            interp_end,
            self.n_out,
        )
        n_avail = int((interp_avail - interp_begin) / delta_t + 1)
        t_avail = np.linspace(
            interp_begin,
            interp_avail,
            n_avail,
        )
        n_aux = int((interp_avail - interp_aux) / delta_t)
        t_aux = np.linspace(
            interp_aux,
            interp_avail,
            n_aux,
        )

        # we only care about Ar and He as tracers
        n_dot_interp_avail = interp(t_avail)[
            [0, 1]
        ]  # shape is (2, n_out) that is two valid samples
        n_dot_interp_aux = interp(t_aux)[[0, 1]]
        n_dot_interp_aux = np.tile(n_dot_interp_aux, (1, 5))
        n_dot_interp = np.concatenate([n_dot_interp_avail, n_dot_interp_aux], axis=1)

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
    self.data_set = torch.utils.data.TensorDataset(self.x, self.t_in, self.y)
    self.train_set, self.test_set = torch.utils.data.random_split(
        self.data_set,
        [
            int(0.8 * len(self.data_set)),
            len(self.data_set) - int(0.8 * len(self.data_set)),
        ],
    )


data_cap.setup = MethodType(fake_setup, data_cap)

data_reac_inlet = RTDDataModule(
    batch_size=32,
    data_file=Path(
        r"D:\Users\Hannes\Documents\Python Code\nRTD\Experiments\HSA_002\data\MGA-E-11032025_120ml-min_1atm_ReactorInlet_MS.npz"
    ),
    switching_times=switching_times_reac_inlet,
    t_range_in=t_in,
    n_in=n_in,
    t_range_out=t_out,
    n_out=n_out,
    switch_delay=1.0,
)

data_cap.setup("fit")
print(data_cap.x.shape)
print(data_cap.t_in.shape)
print(data_cap.y.shape)
print(data_cap.t_out.shape)

index = 5
plt.plot(data_cap.t_in[index], data_cap.x[index].squeeze(), ls="--")
plt.plot(data_cap.t_in[index + 2], data_cap.x[index + 2].squeeze(), ls="--")
plt.plot(data_cap.t_out[index], data_cap.y[index].squeeze(), c="C0", markersize=0.5)
plt.plot(
    data_cap.t_out[index + 2], data_cap.y[index + 2].squeeze(), c="C1", markersize=0.5
)
plt.show()


model = RTDModule(
    kernel_sizes=n_kernels,
    kernel_times=t_kernels,
    learning_rate=4e-3,
    use_scheduler=False,
    scheduler_kwargs={"factor": 0.6, "patience": 2000},
)

for conv_layer in model.net.conv_layers:
    conv_layer.set_to_direct_response()
    E = conv_layer.E.detach().numpy()
    t_kernel = conv_layer.t_kernel.detach().numpy()
    print(np.trapz(E, t_kernel))

model.net.conv_layers[0].freeze()
print(model.net.conv_layers[0].E)
print(model.net.conv_layers[1].E)

my_logger = WandbLogger(log_model=True)
lr_callback = LearningRateMonitor()

trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=20_000,
    enable_progress_bar=True,
    logger=my_logger,
    callbacks=[lr_callback],
)
trainer.fit(model, data_cap)

## prepare the plots
pred_y, pred_t = model(data_cap.x, data_cap.t_in)
pred_t = pred_t.detach().numpy().squeeze()
pred_y = pred_y.detach().numpy().squeeze()

fig = plt.figure()
index = 0
plt.plot(data_cap.t_in[index], data_cap.x[index].squeeze(), ls="--")
plt.plot(data_cap.t_in[index + 2], data_cap.x[index + 2].squeeze(), ls="--")
plt.plot(
    data_cap.t_out[index],
    data_cap.y[index].squeeze(),
    "o",
    c="C0",
    markersize=0.5,
)
plt.plot(
    data_cap.t_out[index + 2],
    data_cap.y[index + 2].squeeze(),
    "o",
    c="C1",
    markersize=0.5,
)
index = 1
plt.plot(data_cap.t_in[index], data_cap.x[index].squeeze(), ls="--", c="C2")
plt.plot(
    data_cap.t_in[index + 2],
    data_cap.x[index + 2].squeeze(),
    ls="--",
    c="C3",
)
plt.plot(
    data_cap.t_out[index],
    data_cap.y[index].squeeze(),
    "o",
    c="C2",
    markersize=0.5,
)
plt.plot(
    data_cap.t_out[index + 2],
    data_cap.y[index + 2].squeeze(),
    "o",
    c="C3",
    markersize=0.5,
)

index = 0
plt.plot(pred_t[index, :], pred_y[index, :], c="C0")
plt.plot(pred_t[index + 2].squeeze(), pred_y[index + 2].squeeze(), c="C1")
index = 1
plt.plot(pred_t[index, :], pred_y[index, :], c="C2")
plt.plot(pred_t[index + 2].squeeze(), pred_y[index + 2].squeeze(), c="C3")
plt.xlabel("t / s")
plt.ylabel("x / 1")
plt.twinx()
ax = plt.gca()
plt.ylabel("E / s^{-1}")
ax.plot(
    model.net.conv_layers[1].t_kernel.detach().numpy(),
    model.net.E[1],
    color="hotpink",
)
wandb.log({"RTD_Plot_cap_img": wandb.Image(fig)})
wandb.log({"RTD_Plot_cap_fig": fig})


model.net.conv_layers[1].freeze()
model.net.conv_layers[0].unfreeze()


model.configure_optimizers()
trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=20_000,
    enable_progress_bar=True,
    logger=my_logger,
    callbacks=[lr_callback],
)
trainer.fit(model, data_reac_inlet)

## prepare the plots
pred_y, pred_t = model(data_reac_inlet.x, data_reac_inlet.t_in)
pred_t = pred_t.detach().numpy().squeeze()
pred_y = pred_y.detach().numpy().squeeze()

fig = plt.figure()
index = 0
plt.plot(data_reac_inlet.t_in[index], data_reac_inlet.x[index].squeeze(), ls="--")
plt.plot(
    data_reac_inlet.t_in[index + 2], data_reac_inlet.x[index + 2].squeeze(), ls="--"
)
plt.plot(
    data_reac_inlet.t_out[index],
    data_reac_inlet.y[index].squeeze(),
    "o",
    c="C0",
    markersize=0.5,
)
plt.plot(
    data_reac_inlet.t_out[index + 2],
    data_reac_inlet.y[index + 2].squeeze(),
    "o",
    c="C1",
    markersize=0.5,
)
index = 1
plt.plot(
    data_reac_inlet.t_in[index], data_reac_inlet.x[index].squeeze(), ls="--", c="C2"
)
plt.plot(
    data_reac_inlet.t_in[index + 2],
    data_reac_inlet.x[index + 2].squeeze(),
    ls="--",
    c="C3",
)
plt.plot(
    data_reac_inlet.t_out[index],
    data_reac_inlet.y[index].squeeze(),
    "o",
    c="C2",
    markersize=0.5,
)
plt.plot(
    data_reac_inlet.t_out[index + 2],
    data_reac_inlet.y[index + 2].squeeze(),
    "o",
    c="C3",
    markersize=0.5,
)

index = 0
plt.plot(pred_t[index, :], pred_y[index, :], c="C0")
plt.plot(pred_t[index + 2].squeeze(), pred_y[index + 2].squeeze(), c="C1")
index = 1
plt.plot(pred_t[index, :], pred_y[index, :], c="C2")
plt.plot(pred_t[index + 2].squeeze(), pred_y[index + 2].squeeze(), c="C3")
plt.xlabel("t / s")
plt.ylabel("x / 1")
plt.twinx()
ax = plt.gca()
plt.ylabel("E / s^{-1}")
ax.plot(
    model.net.conv_layers[0].t_kernel.detach().numpy(),
    model.net.E[0],
    color="goldenrod",
)
ax.plot(
    model.net.conv_layers[1].t_kernel.detach().numpy(),
    model.net.E[1],
    color="hotpink",
)
wandb.log({"RTD_Plot_img": wandb.Image(fig)})
wandb.log({"RTD_Plot_fig": fig})
plt.show()
