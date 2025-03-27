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

argparser = argparse.ArgumentParser()
argparser.add_argument("--n_workers", type=int, default=4)

### Constants
# givens
t_out = (0, 45)
n_out = 181  # 4/s * 45s +1
delta_t_out = (t_out[1] - t_out[0]) / (n_out - 1)

switching_periods = np.array(
    [
        45.18,
        45.28,
        45.12,
        45.11,
        45.20,
        45.21,
        45.25,
        45.08,
        45.08,
        45.00,
    ]
)

switching_times = np.flip(np.cumsum(-switching_periods))
switching_times = np.append(switching_times, 0)


def train():
    run = wandb.init()

    t_kernel = (0, wandb.config.t_kernel_max)
    n_kernel = int(((t_kernel[1] - t_kernel[0]) / delta_t_out) + 1)
    t_in = (0, t_out[1] - t_kernel[1])
    n_in = int(((t_in[1] - t_in[0]) / delta_t_out) + 1)

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

    data = RTDDataModule(
        batch_size=16,
        data_file=Path(
            r"D:\Users\Hannes\Documents\Python Code\nRTD\Experiments\HSA_001\data\MGA-E-05032025_250ml-min_Analytik_MS.npz"
        ),
        switching_times=switching_times,
        t_range_in=t_in,
        n_in=n_in,
        t_range_out=t_out,
        n_out=n_out,
        switch_delay=0.7,
    )
    model = RTDModule(
        kernel_sizes=[n_kernel],
        kernel_times=[t_kernel],
        learning_rate=wandb.config["learning_rate"],
        use_scheduler=True,
        scheduler_kwargs={
            "factor": wandb.config["lrscheduler_factor"],
            "patience": wandb.config["lrscheduler_patience"],
        },
    )

    model.net.conv_layers[0].set_to_direct_response()

    my_logger = WandbLogger(log_model=True)

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=30_000,
        logger=my_logger,
        enable_progress_bar=False,
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
    plt.plot(data.t_out[index], data.y[index].squeeze(), c="C0")
    plt.plot(data.t_out[index + 2], data.y[index + 2].squeeze(), c="C1")

    plt.plot(pred_t[index, :], pred_y[index, :])
    plt.plot(pred_t[index + 2].squeeze(), pred_y[index + 2].squeeze())

    plt.plot(model.net.conv_layers[0].t_kernel.detach().numpy(), model.net.E[0])

    wandb.log({"RTD_Plot_fig": fig})
    wandb.log({"RTD_Plot_img": wandb.Image(fig)})

    # np.savez(
    #     r"D:\Users\Hannes\Documents\Python Code\nRTD\Experiments\HSA_001\data\MGA-E-05032025_75ml-min_Analytik_MS_E_nRTD.npz",
    #     E=model.net.E[0],
    #     t=model.net.conv_layers[0].t_kernel.detach().numpy(),
    # )


def main(args):
    this_file = Path(__file__)

    wandb.setup()

    sweep_config = {
        "method": "grid",
        "name": this_file.stem,
        "metric": {"goal": "minimize", "name": "test/loss"},
        "parameters": {
            "t_kernel_max": {
                "values": [15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35],
            },
            "learning_rate": {
                "value": 4e-3,
            },
            "lrscheduler_factor": {
                "value": 0.65,
            },
            "lrscheduler_patience": {
                "value": 94,
            },
        },
    }

    project_str = "HSA_MGA_nRTD"
    sweep_id = wandb.sweep(sweep=sweep_config, entity="ice_ulm", project=project_str)

    sweeper = Sweeper(
        project_str=project_str,
        sweep_id=sweep_id,
        train_fn=train,
        n_workers=args.n_workers,
    )

    sweeper.start()


args = argparser.parse_args()
if __name__ == "__main__":
    main(args)
