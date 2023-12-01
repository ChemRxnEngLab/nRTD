from typing import Any
import torch
import lightning.pytorch as pl
from . import rtd_net


class RTDModule(pl.LightningModule):
    def __init__(
        self,
        kernel_size: int,
        padding_mode: str = "replicate",
        n_compartements: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.net = rtd_net.RTDNet(
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            n_compartements=n_compartements,
        )

        self.save_hyperparameters(
            ignore=[
                "padding_mode",
            ],
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        ret_dict = {
            "optimizer": torch.optim.Adam(self.parameters(), lr=1e-5),
        }
        return ret_dict
