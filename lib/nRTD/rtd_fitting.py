from typing import Any, Optional
import torch
import lightning.pytorch as pl
from . import rtd_net


class RTDModule(pl.LightningModule):
    def __init__(
        self,
        kernel_size: int,
        padding_mode: str = "replicate",
        n_compartements: int = 1,
        learning_rate: float = 1e-3,
        use_scheduler: bool = False,
        scheduler_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.net = rtd_net.RTDNet(
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            n_compartements=n_compartements,
        )

        if use_scheduler and scheduler_kwargs is None:
            raise ValueError(
                "scheduler_kwargs must be provided when use_scheduler is True."
            )

        self.use_scheduler = use_scheduler
        self.scheduler_kwargs: dict[str, Any] = scheduler_kwargs  # type: ignore

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
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val/loss", loss)
        # return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
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
