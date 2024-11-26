import matplotlib.pyplot as plt
import numpy.typing as npt
import sys
import os
module_path = os.path.expanduser("lib")
sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import numpy as np
import wandb
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
from nRTD.rtd_fitting_3 import RTDModule
from nRTD.rtd_net_4 import RTDNet
from lightning.pytorch import loggers as pl_loggers
import os
import datetime
import sympy as sp
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch


base_dir = r"D:\Tuana\nRTD\Experiments\Preliminary\Litrature\Laminar_Flow_Model_dynamic\tau_5.0"
save_dir = r"D:\Tuana\nRTD\Experiments\Preliminary/Convolution_script\Convolution_018_Laminar_Model"


n_variations = 30
c_in_list = []
c_out_list = []

for i in range(1, n_variations + 1):
    variation_dir = os.path.join(base_dir, f"variation_{i}")
    t_conv = np.load(os.path.join(variation_dir, "time.npy"))
    c_out = np.load(os.path.join(variation_dir, "concentration.npy"))
    input_func = np.load(os.path.join(variation_dir, "input_function.npy"))

    c_in_list.append(torch.tensor(input_func, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    c_out_list.append(torch.tensor(c_out, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

c_in_train = torch.cat(c_in_list[:-1], dim=0)  
c_out_train = torch.cat(c_out_list[:-1], dim=0)
c_in_test = c_in_list[-1]  
c_out_test = c_out_list[-1]
train_dataset = TensorDataset(c_in_train, c_out_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

test_dataset = TensorDataset(c_in_test, c_out_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model = RTDModule(
    kernel_sizes=[151],
    kernel_times=[(0.0, 75)],
    learning_rate=1e-4,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

wandb_logger = pl_loggers.WandbLogger(project="nRTD", log_model=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=17000,
    logger=wandb_logger,
    deterministic=True,
)


trainer.fit(model, train_loader)
results = trainer.test(model, test_loader)
E_pred = model.net.E[0]
t_E = torch.linspace(0, 75, 151)
E_pred /= E_pred.max()

# Expected E calculation
t_E_np = np.linspace(0, 75, 151)
def expected_formula(t):
    return np.where(t >= 2.5, (2.5**2) / (2 * (t**3)), 0)
E_expected = expected_formula(t_E_np)
E_expected /= E_expected.max()
plt.figure(figsize=(10, 6))
plt.plot(t_E, E_pred.detach().numpy(), label="E Predicted", color="orange")
plt.plot(t_E_np, E_expected, label="E Expected", linestyle="--", color="purple")
plt.xlabel("Time")
plt.ylabel("E")
plt.legend()
plt.title("E Predicted vs. Expected")
plt.savefig(os.path.join(save_dir, "E_pred_vs_expected.png"), dpi=300)
plt.show()
np.save(os.path.join(save_dir, "E_predicted.npy"), E_pred.detach().numpy())
np.save(os.path.join(save_dir, "t_E_predicted.npy"), t_E.numpy())
np.save(os.path.join(save_dir, "E_expected.npy"), E_expected)
np.save(os.path.join(save_dir, "t_E_expected.npy"), t_E_np)

