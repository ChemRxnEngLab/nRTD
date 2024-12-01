import sys

sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")
# sys.path.append("lib")
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
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
from ICIW_Plots import cm2inch

if wandb.run is not None:
    wandb.finish()

n_in_1, n_out_1, n_e_1, = sp.symbols(
    "n_in_1 n_out_1 n_e_1 ", positive=True, real=True
)
t_i, t_o, t_e_1 = sp.symbols(
    "t_i, t_o, t_e_1 ", positive=True, real=True
)
equations = [
    t_i - t_o + t_e_1,
    n_in_1 - n_out_1 + n_e_1 - 1,
    n_in_1 / t_i - n_e_1 / t_e_1,
]
for equation in equations:
    print(equation)
solution_set = sp.solve(
    equations, (n_in_1, n_out_1, n_e_1, t_o), dict=True
)
print(solution_set)
solution_set = sp.solve(
    equations,
    n_in_1,
    # n_out_1,
    n_e_1,
    # t_in,
    #t_o,
    # t_e_1,
    dict=True,
)
solution = solution_set[0]
print(len(solution_set))
for _, val in solution.items():
    print(val)
sub_dict = {
    # n_in_1,
    # n_out_1,
    n_out_1: 500,
    # n_e_1,
    t_i: 20,
    t_o:41,
    t_e_1:21,
}
result_dict = {}

for key, value in solution.items():
    print(f"{key} = {value}")
    evaluated_value = value.subs(sub_dict)
    if evaluated_value.free_symbols:
        print(
            f"Cannot fully evaluate {key}: remaining symbols {evaluated_value.free_symbols}"
        )
    else:
        result_dict[key] = (float(sp.N(evaluated_value)))
        print(f"Updated {key}: {result_dict[key]}")
print(result_dict)
n_in_1 = int(sp.floor(result_dict.get(n_in_1, None))) 
n_1_out = result_dict.get(n_out_1, sub_dict.get(n_out_1))
n_e_1 = int(ceiling(result_dict.get(n_e_1, None))) 
t_o = result_dict.get(t_o, sub_dict.get(t_o))
t_i = sub_dict.get(t_i)
t_e_1 = sub_dict.get(t_e_1)
t_e = t_e_1
print("n_in_1 =", n_in_1)
print("n_1_out =", n_1_out)
print("n_e_1 =", n_e_1)
print("t_o =", t_o)
print("t_i =", t_i)
print("t_e_1 =", t_e_1)

t_input = torch.linspace(0, t_i, n_in_1)
c_in = torch.zeros((20, 1, n_in_1))
c_in[::2, :, t_input > 1] = 0.1
c_in[1::2, :, t_input < 1] = 0.1
file_numbers = range(1, 21)
c_out_list = []
t_conv_list = []
# file_numbers = range(1, 21, 2)

for i, file_num in enumerate(file_numbers):
    t_conv_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_t_processed.npy"
    c_out_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_x_processed.npy"

    print(f"Processing files: {t_conv_path}, {c_out_path}")

    t_conv = (
        torch.tensor(np.load(t_conv_path), dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    c_out = (
        torch.tensor(np.load(c_out_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    c_out_list.append(c_out)
    t_conv_list.append(t_conv)
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)
plt.show()
print(f"c_in size: {c_in.size()}")
print(f"c_out size: {c_out.size()}")
print(f"t_conv size: {t_conv.size()}")

model = RTDModule(
    kernel_sizes=[n_e_1],
    kernel_times=[(0.0, t_e_1)],
    learning_rate=1e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)
c_conv = model(c_in)
E = model.net.E[0]
t_E =torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
j = 2

plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label=f"Exp_{file_numbers[i]}",
        color="green",
    )

plt.plot(
    t_conv[j, 0, :].numpy(),
    c_conv[j, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.legend()
plt.xlim((0, 10))
plt.ylim((0, 0.2))
plt.show()

print(model(c_in).size())

ds = TensorDataset(c_in, c_out)
# split the dataset into train and test
# train_ds, test_ds = random_split(ds, [0.8, 0.2])

# train_dl = DataLoader(train_ds, batch_size=1)
# test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

dl = DataLoader(ds, batch_size=20, shuffle=True)

# set up the logger
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000,
    logger=wandb_logger, deterministic=True
)

# trainer.fit(model, train_dl)
trainer.fit(model, dl)
# adds an epoch at the end to calculate the final loss at the traineing end
# trainer.test(model, test_dl)
trainer.test(model, dl)  ### check it maybe you will see changes??

##################
# Postprocessing #
##################

c_conv = model(c_in)
E = model.net.E[0]
t_E =torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

# Iterate over the loaded files and plot them
for i in range(c_out.size(1)):
    plt.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label=f"Exp_{file_numbers[i]}",
        color="green",
    )

plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.xlim((0, 51))
plt.ylim((0, 0.01))
plt.legend()

# fig = plt.gcf()
# wandb.log({"RTD_Plot": fig})
# wandb.log({"RTD_Plot": wandb.Image(fig)})


#wandb.finish()

# plt.savefig("Figure_C_004_H_080")
plt.show()


