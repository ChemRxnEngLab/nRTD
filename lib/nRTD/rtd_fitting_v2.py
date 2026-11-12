"""Version 2 of rtd_fitting: RTDModule on rtd_net_v2.

Differences to rtd_fitting.RTDModule:
  - `positivity="softmax"` (default): the kernels are fitted in logit space (rtd_net_v2.SoftmaxKernel), E ≥ 0 and
    Σ E·Δt = 1 hold by construction; no clamping.
  - `positivity="clamp"`: free weights, clamped to E ≥ 0 AFTER every optimizer step (`on_train_batch_end`).
    rtd_fitting clamps before each step, so the weights after the last step are never clamped and the stored kernels
    can hold negative values (CCA_004 He BU E_down: −0.043 in Σ E·Δt). For a feasible start both orders give the
    same sequence of losses; only the final kernels differ.
  - `adam_eps` is exposed, default 1e-16 for softmax, 1e-8 (Adam's default) for clamp: the gradients of the logits
    are small (CCA losses O(1e-7-1e-6); median √v at the end of a stage 1e-13 to 1e-11), so eps 1e-8 damps the logit
    updates and makes the result depend on the learning rate (CCA_005 docstring, scratchpad scan 2026-11-02).
The data module is unchanged: `RTDDataModule` is rtd_fitting's.
"""

from typing import Any, Optional
import torch
import lightning.pytorch as pl
from . import rtd_net_v2 as rtd_net
from .rtd_fitting import RTDDataModule  # noqa: F401  (unchanged, re-exported)


class RTDModule(pl.LightningModule):
    def __init__(
        self,
        kernel_sizes: list[int],
        kernel_times: Optional[list[tuple[float, float]]] = None,
        learning_rate: float = 1e-3,
        use_scheduler: bool = False,
        scheduler_kwargs: Optional[dict[str, Any]] = None,
        positivity: str = "softmax",
        floor: float = 1e-12,
        adam_eps: Optional[float] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.kernel_sizes = kernel_sizes
        self.kernel_times = kernel_times
        self.positivity = positivity
        if adam_eps is None:
            adam_eps = 1e-16 if positivity == "softmax" else 1e-8
        self.adam_eps = adam_eps
        self.net = rtd_net.RTDNet(
            kernel_sizes=self.kernel_sizes,
            kernel_times=self.kernel_times,
            positivity=positivity,
            floor=floor,
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
        y_hat, _ = self(X, t)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # after the optimizer step: the kernels are feasible whenever training stops
        self.net.clamp_()

    def validation_step(self, batch, batch_idx):
        X, t, y = batch
        y_hat, _ = self(X, t)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        X, t, y = batch
        y_hat, _ = self(X, t)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test/loss", loss)

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore
        _optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, eps=self.adam_eps
        )
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
