import matplotlib.pyplot as plt
import numpy.typing as npt
import sys
import os
module_path = r"D:\Tuana\nRTD\lib"
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

if wandb.run is not None:
    wandb.finish()
    
save_dir=r"D:\Tuana\nRTD\Experiments\Preliminary\2_nd_Layer_exp\0001\H_185"
learning_rate=1e-3
learning_rate_2=1e-1
n_in_1, n_out_1, n_out_2, n_e_1, n_e_2 = sp.symbols("n_in_1 n_out_1 n_out_2 n_e_1 n_e_2", positive=True, real=True)
t_1, t_lam, t_adl, t_e_1, t_e_2 = sp.symbols("t_1, t_lam, t_adl, t_e_1, t_e_2", positive=True, real=True)
equations = [
    t_1 - t_lam + t_e_1,
    n_in_1 - n_out_1 + n_e_1 -1,
    n_in_1 / t_1 - n_e_1 / t_e_1,
    t_lam - t_adl + t_e_2,
    n_out_1 - n_out_2 + n_e_2 - 1,
    n_e_2 * t_lam - n_out_1 * t_e_2]
for equation in equations:
    print(equation)
solution_set = sp.solve(equations,(n_in_1, n_out_1, n_out_2, n_e_1, n_e_2, t_lam, t_e_2),dict=True)
print(solution_set) 
solution_set = sp.solve(
    equations,
    n_in_1,
    #n_out_1,
    n_out_2,
    n_e_1,
    n_e_2,
    # t_1,
    t_lam,
    # t_adl,
    # t_e_1,
    t_e_2,
    dict=True,
)
solution = solution_set[0]
print(len(solution_set))
for _, val in solution.items():
    print(val)
sub_dict = {
    #n_in_1,
    #n_out_1,
    n_out_1: 200,
    # n_e_1,
    # n_e_2,
    t_1:10,
    # t_lam,
    t_adl:51,
    t_e_1:11,
    #t_e_2,
}
result_dict = {}

for key, value in solution.items():
    print(f"{key} = {value}")
    evaluated_value = value.subs(sub_dict)
    if evaluated_value.free_symbols:
        print(f"Cannot fully evaluate {key}: remaining symbols {evaluated_value.free_symbols}")
    else:
        result_dict[key] = round(float(sp.N(evaluated_value)))
        print(f"Updated {key}: {result_dict[key]}")
print(result_dict)
n_1_in = result_dict.get(n_in_1, None)
n_1_out = result_dict.get(n_out_1, sub_dict.get(n_out_1))
n_e_1 = ceiling(result_dict.get(n_e_1, None))
n_1_E=n_e_1 
n_2_in = n_1_out
n_2_out = result_dict.get(n_out_2)
n_e_2 = result_dict.get(n_e_2)
t_lam = result_dict.get(t_lam, sub_dict.get(t_lam))
t_adl = sub_dict.get(t_adl)
t_1 = sub_dict.get(t_1)
t_e_1 = sub_dict.get(t_e_1)
t_e=t_e_1
t_e_2 = result_dict.get(t_e_2)
print("n_1_in =", n_1_in)
print("n_1_out =", n_1_out)
print("n_e_1 =", n_e_1)
print("n_2_in =", n_2_in)
print("n_2_out =", n_2_out)
print("n_e_2 =", n_e_2)
print("t_lam =", t_lam)
print("t_adl =", t_adl)
print("t_1 =", t_1)
print("t_e_1 =", t_e_1)
print("t_e_2 =", t_e_2)
c_conv_results = {}
n_disc = n_in_1
t_input = torch.linspace(0, t_1, n_1_in)
c_in = torch.zeros((9, 1, n_1_in))
c_in[::2, :, t_input > 1] =2.53669964e-02
c_in[1::2, :, t_input > 1] =2.53669964e-02
c_out_list = []
t_conv_list = []
file_numbers = range(1, 21)
excluded_files = {19,20}
filtered_file_numbers = [num for num in file_numbers if num not in excluded_files]


for i, file_num in enumerate(filtered_file_numbers):
    if file_num % 2 == 0:  # Skip odd-numbered indices
        continue 
    #t_conv_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1_001/TOA_MGA_20231013_007_{file_num:06d}_t_processed.npy"
    #c_out_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1_001/TOA_MGA_20231013_007_{file_num:06d}_x_processed.npy"
    t_conv_path= rf"D:\Tuana\nRTD\Data\0001\C_001\H_185_C1\S_007_C1\TOA_MGA_20231013_007_{file_num:06d}_t_processed_norm_reg.npy"
    c_out_path = rf"D:\Tuana\nRTD\Data\0001\C_001\H_185_C1\S_007_C1\TOA_MGA_20231013_007_{file_num:06d}_x_processed_norm_reg.npy"

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
plt.show()

print(model(c_in).size())

ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=8000,
    logger=wandb_logger, deterministic=True
)
trainer.fit(model, dl)
trainer.test(model, dl) 
c_conv = model(c_in)
c_conv_results[n_1_out] = c_conv.detach().numpy()
E = model.net.E[0]
t_E =torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
# E=E/E.max()
plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

import matplotlib.pyplot as plt

# Loop over all experiments and plot their profiles
for idx, exp_num in enumerate(filtered_file_numbers):
    plt.figure(figsize=(10, 6))
    plt.plot(t_input.numpy(), c_in[idx, 0, :].numpy(), label="Input (SF)", color="blue")
    
    # Plot each experimental profile
    for i in range(c_out.size(1)):
        plt.plot(
            t_conv[idx, i, :].numpy(),
            c_out[idx, i, :].numpy(),
            label=f"Exp_{filtered_file_numbers[i]}",
            color="green",
            alpha=0.7,
        )
    
    # Plot the predicted profile
    plt.plot(
        t_conv[idx, 0, :].numpy(),
        c_conv[idx, 0, :].detach().numpy(),
        label="Predicted",
        color="red",
    )
    
    # Plot the kernel profile
    plt.plot(t_E, E, label="Kernel (E)", color="orange")

    # Customize plot
    plt.title(f"Profile Comparison for Experiment {exp_num}")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Show or save the plot
    plt.show()
