import sys

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

torch.set_default_dtype(torch.float64)

from nRTD.rtd_fitting_5 import RTDModule, RTDDataModule
from SweepRunner import Sweeper
import matplotlib.pyplot as plt

plt.style.use("ICIWstyle")

argparser = argparse.ArgumentParser()
argparser.add_argument("--n_workers", type=int, default=4)

### Constants
# givens
t_out = (0, 120)
n_out = 4 * (t_out[1] - t_out[0]) + 1  # 4/s * 45s +1
delta_t_out = (t_out[1] - t_out[0]) / (n_out - 1)

t_offset = 7.4 - 1
t_halfperiod = t_out[1] - t_out[0]

switching_times = np.full((20,), t_halfperiod)
switching_times = np.insert(switching_times, 0, t_offset)
switching_times = np.cumsum(switching_times)


t_kernel = (0, 80)
n_kernel = int(((t_kernel[1] - t_kernel[0]) / delta_t_out) + 1)
t_in = (0, t_out[1] - t_kernel[1])
n_in = int(((t_in[1] - t_in[0]) / delta_t_out) + 1)

wandb.init(project="HSA_MGA_nRTD", entity="ice_ulm")

wandb.config.update(
    {
        "t_out": t_out,
        "n_out": n_out,
        "delta_t_out": delta_t_out,
        "t_in": t_in,
        "n_in": n_in,
        "t_kernel": t_kernel,
        "n_kernel": n_kernel,
    },
)

print(f"input signal: {t_in},  discretized to {n_in} points")
print(f"kernel: {t_kernel},  discretized to {n_kernel} points")
print(f"output signal: {t_out},  discretized to {n_out} points")

data = RTDDataModule(
    batch_size=32,
    data_file=Path(
        r"D:\Users\Hannes\Documents\Python Code\nRTD\Experiments\HSA_004\data\MGA-E-13032025_30ml-min_1atm_heated_TotalSystem.npz"
    ),
    switching_times=switching_times,
    t_range_in=t_in,
    n_in=n_in,
    t_range_out=t_out,
    n_out=n_out,
    switch_delay=1.0,
)

data.setup("fit")
print(data.x.shape)
print(data.t_in.shape)
print(data.y.shape)
print(data.t_out.shape)

index = 5
plt.plot(data.t_in[index], data.x[index].squeeze(), ls="--")
plt.plot(data.t_in[index + 2], data.x[index + 2].squeeze(), ls="--")
plt.plot(data.t_out[index], data.y[index].squeeze(), c="C0", markersize=0.5)
plt.plot(data.t_out[index + 2], data.y[index + 2].squeeze(), c="C1", markersize=0.5)
plt.show()


model = RTDModule(
    kernel_sizes=[n_kernel],
    kernel_times=[t_kernel],
    learning_rate=4e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.6, "patience": 100},
)

model.net.conv_layers[0].set_to_direct_response()
my_logger = WandbLogger(log_model=True)

trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=10_000,
    enable_progress_bar=True,
    logger=my_logger,
)
trainer.fit(model, data)
trainer.test(model, data)

## prepare the plots
pred_y, pred_t = model(data.x, data.t_in)
pred_t = pred_t.detach().numpy().squeeze()
pred_y = pred_y.detach().numpy().squeeze()

fig = plt.figure()
index = 0
plt.plot(data.t_in[index], data.x[index].squeeze(), ls="--")
plt.plot(data.t_in[index + 2], data.x[index + 2].squeeze(), ls="--")
plt.plot(data.t_out[index], data.y[index].squeeze(), "o", c="C0", markersize=0.5)
plt.plot(
    data.t_out[index + 2], data.y[index + 2].squeeze(), "o", c="C1", markersize=0.5
)
index = 1
plt.plot(data.t_in[index], data.x[index].squeeze(), ls="--", c="C2")
plt.plot(data.t_in[index + 2], data.x[index + 2].squeeze(), ls="--", c="C3")
plt.plot(data.t_out[index], data.y[index].squeeze(), "o", c="C2", markersize=0.5)
plt.plot(
    data.t_out[index + 2], data.y[index + 2].squeeze(), "o", c="C3", markersize=0.5
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
wandb.log({"RTD_Plot_img": wandb.Image(fig)})
wandb.log({"RTD_Plot_fig": fig})
plt.show()


# np.savez(
#     r"D:\Users\Hannes\Documents\Python Code\nRTD\Experiments\HSA_001\data\MGA-E-05032025_75ml-min_Analytik_MS_E_nRTD.npz",
#     E=model.net.E[0],
#     t=model.net.conv_layers[0].t_kernel.detach().numpy(),
# )
