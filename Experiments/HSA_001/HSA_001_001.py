import sys

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch.utils
import torch.utils.data

sys.path.append(r"D:\Users\Hannes\Documents\Python Code\nRTD\lib")
from nRTD.rtd_net import RTDNet
from nRTD.rtd_fitting import RTDModule
import lightning.pytorch as pl
from pathlib import Path
import numpy as np
import scipy
import torch

torch.set_default_dtype(torch.float64)


class RTDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_file: Path,
        switching_periods: list[float],
        t_range_in: tuple[float, float],
        n_in: int,
        t_range_out: tuple[float, float],
        n_out: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_file = data_file
        # calculate the absolute times from the relative times between two switching events
        self.switching_periods = np.asarray(switching_periods)
        self.switching_times = np.flip(np.cumsum(-self.switching_periods))
        self.switching_times = np.append(self.switching_times, 0)
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
        self.switch_delay = 1.0  # delay between the button click and the actual pshhht

    def setup(self, stage: str):
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


def main():
    import matplotlib.pyplot as plt

    plt.style.use("ICIWstyle")

    switching_periods = np.array(
        [
            45.15,
            45.06,
            44.96,
            45.05,
            45.05,
            45.05,
            45.21,
            45.11,
            45.08,
            45.06,
        ]
    )

    ### the hyperparameters
    # givens
    t_out = (0, 45)
    n_out = 181  # 4/s * 45s +1
    # wishes
    t_kernel = (0, 25)
    # consequences
    delta_t_out = (t_out[1] - t_out[0]) / (n_out - 1)
    n_kernel = int(((t_kernel[1] - t_kernel[0]) / delta_t_out) + 1)
    t_in = (0, t_out[1] - t_kernel[1])
    n_in = int(((t_in[1] - t_in[0]) / delta_t_out) + 1)

    print(t_in, n_in)
    print(t_out, n_out)
    print(t_kernel, n_kernel)

    data = RTDDataModule(
        batch_size=1,
        data_file=Path(
            r"D:\Users\Hannes\Documents\Python Code\nRTD\Experiments\HSA_001\data\MGA-E-05032025_75ml-min_Analytik_MS.npz"
        ),
        switching_periods=switching_periods,
        t_range_in=t_in,
        n_in=n_in,
        t_range_out=t_out,
        n_out=n_out,
    )
    data.setup("fit")
    print(data.x.shape)
    print(data.t_in.shape)
    print(data.y.shape)
    print(data.t_out.shape)

    index = 5
    plt.plot(data.t_in[index], data.x[index].squeeze(), ls="--")
    plt.plot(data.t_in[index + 2], data.x[index + 2].squeeze(), ls="--")
    plt.plot(data.t_out[index], data.y[index].squeeze(), c="C0")
    plt.plot(data.t_out[index + 2], data.y[index + 2].squeeze(), c="C1")

    model = RTDModule(
        kernel_sizes=[n_kernel],
        kernel_times=[t_kernel],
        learning_rate=0.001,
        use_scheduler=True,
        scheduler_kwargs={"factor": 0.5, "patience": 100},
    )

    trainer = pl.Trainer(max_epochs=5000)
    trainer.fit(model, data)

    pred_y, pred_t = model(data.x, data.t_in)
    pred_t = pred_t.detach().numpy().squeeze()
    pred_y = pred_y.detach().numpy().squeeze()
    print(pred_t.shape)
    print(pred_y.shape)
    plt.plot(pred_t[index, :], pred_y[index, :])
    plt.plot(pred_t[index + 2].squeeze(), pred_y[index + 2].squeeze())

    np.savez(
        r"D:\Users\Hannes\Documents\Python Code\nRTD\Experiments\HSA_001\data\MGA-E-05032025_75ml-min_Analytik_MS_E_nRTD.npz",
        E=model.net.E[0],
        t=model.net.conv_layers[0].t_kernel.detach().numpy(),
    )

    plt.plot(model.net.conv_layers[0].t_kernel.detach().numpy(), model.net.E[0])
    plt.show()


if __name__ == "__main__":
    main()
