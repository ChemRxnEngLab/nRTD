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

if wandb.run is not None:
    wandb.finish()
    
#save_dir=r"D:\Tuana\nRTD\Experiments\Preliminary\2_nd_Layer_exp\0001\H_185"
learning_rate=1e-3
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
c_in = torch.zeros((20, 1, n_1_in))
c_in[::2, :, t_input > 1] = 0.025
c_in[1::2, :, t_input < 1] = 0.025
file_numbers = range(1, 21)
c_out_list = []
t_conv_list = []


for i, file_num in enumerate(file_numbers):
    t_conv_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1_001/TOA_MGA_20231013_007_{file_num:06d}_t_processed.npy"
    c_out_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1_001/TOA_MGA_20231013_007_{file_num:06d}_x_processed.npy"
    #t_conv_path= r"D:\Tuana\nRTD\Experiments\Data\0001\C_001\H_185_C1\S_007_C1_001\TOA_MGA_20231013_007_{file_num:06d}_t_processed_2nlayer_200.npy"
    #c_out_path =r"D:\Tuana\nRTD\Experiments\Data\0001\C_001\H_138_C1\S_007_C1_001\TOA_MGA_20231013_007_{file_num:06d}_x_processed_2nlayer_200.npy"
    
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
dl = DataLoader(ds, batch_size=20, shuffle=True)
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=2,
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
plt.xlim((0, 10))
plt.ylim((0, 0.2))
plt.legend()
wandb.finish()
current_date = datetime.datetime.now().strftime("%Y%m%d")
#plt.savefig(os.path.join(save_dir,f"Figure_C_001_H_185_C1{current_date}.png"), dpi=300)
plt.show()

###Second CNN
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True,
    reinit=True
)
c_conv_2_results = {}
c_out_list_2 = []
t_conv_list_2 = []
t_2_in=torch.tensor(t_conv).float()
c_2_in=torch.tensor(c_out).float()
print("c_in",c_out.shape)

for i, file_num in enumerate(file_numbers):
    t_conv_path_2 = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_185_C2/S_014_C2/TOA_MGA_20231020_014_{file_num:06d}_t_processed.npy"
    c_out_path_2 = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_185_C2/S_014_C2/TOA_MGA_20231020_014_{file_num:06d}_x_processed.npy"
    # t_conv_path_2= r"D:\Tuana\nRTD\Experiments\Data\0001\C_002\H_185_C2\S_014_C2\TOA_MGA_20231020_014_{file_num:06d}_t_processed.npy"
    # c_out_path_2 =r"D:\Tuana\nRTD\Experiments\Data\0001\C_002\H_185_C2\S_014_C2\TOA_MGA_20231020_014_{file_num:06d}_x_processed.npy"
    
    print(f"Processing files: {t_conv_path_2}, {c_out_path_2}")

    t_conv_2 = (
        torch.tensor(np.load(t_conv_path_2), dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    c_out_2 = (
        torch.tensor(np.load(c_out_path_2), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    c_out_list_2.append(c_out_2)
    t_conv_list_2.append(t_conv_2)
c_out_2 = torch.cat(c_out_list_2, dim=0)
t_conv_2 = torch.cat(t_conv_list_2, dim=0)

print(c_out_2.shape)
print(c_2_in.shape)

model_2 = RTDModule(
    kernel_sizes=[n_e_2],
    kernel_times=[(0.0, t_e_2)],
    learning_rate=learning_rate,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)
c_conv_2 = model_2(c_2_in)
E_2 = model_2.net.E[0]
t_E_2 = torch.linspace(0, t_e_2, int(model_2.kernel_sizes[0]))
ds_2 =TensorDataset(c_2_in.float(), c_out_2.float())
dl_2 = DataLoader(ds_2, batch_size=1, shuffle=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=2,
    logger=wandb_logger,
    deterministic=True,
)
# trainer = pl.Trainer(
#     accelerator="auto",
#     max_epochs=epoch,
#     deterministic=True,
# )
trainer.fit(model_2, dl_2)
trainer.test(model_2, dl_2)
E_2 = model_2.net.E[0]
c_conv_2 = model_2(c_2_in)
c_conv_2_results[n_2_out] = c_conv_2.detach().numpy()
#E = model_2.net.E[0]


fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(10, 8))
plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Input",
    color="orange",
)
for i in range(c_out.size(1)):
    plt.plot(
        t_conv_2[0, i, :].numpy(),
        c_out_2[0, i, :].numpy(),
        label=f"Exp_{file_numbers[i]}",
        color="green",
    )

plt.plot(
    t_conv_2[0, 0, :].numpy(),
    c_conv_2[0, 0, :].detach().numpy(),
    label="Predicted_2",
    color="red",
)
ax1.plot(t_E_2, E_2, label="E_predict", color="purple", linestyle="--")
print(t_E_2.shape)
print(E_2.shape)
ax1.set_xlim((0, 51))
ax1.set_ylim((0, 0.2))
ax1.set_ylabel('Concentration')
ax1.set_xlabel('Time')
current_date = datetime.datetime.now().strftime("%Y%m%d")
#plt.savefig(os.path.join(save_dir,f"Figure_C_002_H_185_C2{current_date}.png"), dpi=300)
current_date = datetime.datetime.now().strftime("%Y%m%d")
#plt.savefig(os.path.join(save_dir,f"Figure_C_002_H_185_C2{current_date}.png"), dpi=300)
plt.show()
plt.show()

# import ICIW_Plots.colors as ICIWcolors
# from ICIW_Plots.figures import Elsevier_Sizes
# import datetime
# import ICIW_Plots.colors as ICIWcolors
# from ICIW_Plots.figures import Elsevier_Sizes, ACS_Sizes
# from ICIW_Plots import make_square_ax, cm2inch
# from ICIW_Plots import make_rect_ax
# plt.style.use("ICIWstyle")
# fig = plt.figure(figsize=(Elsevier_Sizes.single_column["in"], 12 * cm2inch))
# ax = make_rect_ax(
#     fig,
#     ax_width=7.3 * cm2inch,
#     ax_height=5 * cm2inch,
#     # left_h=0.2,  # These arguments control the spacing of the axis
#     # bottom_v=0.2, # not supplying them wil place the axes in the middle of the figure
#     xlabel=r"$t$ / $s$",
#     ylabel=r"$E$ / $1$"
# )
# ax.plot(t_2_in.squeeze().numpy(), c_2_in.squeeze().numpy(), )
# ax.plot(t_1_in_reshaped, c_1_in_reshaped, label=r"$x_{0(t)}$",color=ICIWcolors.CERULEAN)
# ax.plot(t_conv_reshaped, c_conv_reshaped, label="$\hat{x}_{1(t)}$",color="purple",linestyle="--")
# ax.plot(t_conv_reshaped, c_out_l_reshaped, label=r"$x_{1(t)}$",color=ICIWcolors.DRAB)
# ax.plot(t_out_l_a.squeeze().numpy(), c_conv_2.detach().squeeze().numpy(), label="$\hat{x}_{2(t)}$", color="black", linestyle="--")
# ax.plot(t_out_l_a.squeeze().numpy(), c_out_l_a.squeeze().numpy(), label=r"$x_{2(t)}$", color=ICIWcolors.FLAME)
# ax1.set_xlim((0, 30))
# ax1.set_ylim((-0.1, 1.1))
# current_date = datetime.datetime.now().strftime("%Y%m%d")
# #plt.savefig(f"2nd_Layer_Ch_and_Adl_{current_date}.png", dpi=300)
# plt.show()

# #### General Plotting
# E_Adler_normalized = E_Adler / np.max(E_Adler)
# E_learned_1 = model.net.E[0] if isinstance(model.net.E[0], np.ndarray) else model.net.E[0].numpy()
# E_learned_2 = model_2.net.E[0] if isinstance(model_2.net.E[0], np.ndarray) else model_2.net.E[0].numpy()
# E_learned_1 /= np.max(E_learned_1)
# E_learned_2 /= np.max(E_learned_2)
# t_adler = t_values_Adler if isinstance(t_values_Adler, np.ndarray) else t_values_Adler.numpy()
# t_learned_1 = np.linspace(0, t_e, len(E_learned_1))
# t_learned_2 = np.linspace(0, t_e_2, len(E_learned_2))

# plt.style.use("ICIWstyle")
# fig = plt.figure(figsize=(Elsevier_Sizes.single_column["in"], 12 * cm2inch))
# ax = make_rect_ax(
#     fig,
#     ax_width=7.3 * cm2inch,
#     ax_height=5 * cm2inch,
#     # left_h=0.2,  # These arguments control the spacing of the axis
#     # bottom_v=0.2, # not supplying them wil place the axes in the middle of the figure
#     xlabel=r"$t$ / $s$",
#     ylabel=r"$E$ / $1$"
# )
# ax.plot(t_learned_1, E_learned_1, label="$\hat{E}_{1(t)}$", color="purple")
# ax.plot(t_learned_2, E_learned_2, label="$\hat{E}_{2(t)}$", color=ICIWcolors.FLAME)
# ax1.set_xlim((0, 30))
# ax1.set_ylim((-0.1, 1.1))
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(t_learned_1, E_learned_1, label="$E_1$", color="purple")
# plt.plot(t_learned_2, E_learned_2, label="$E_2$", color="red")
# # plt.plot(t_l, E_laminar, label="E Laminar Layer 1", color="blue")
# #plt.plot(t_l_a, E_laminar_a_normalized, label="E Laminar Layer 2", color="green")
# # plt.plot(t_adler, E_Adler_normalized, label="E Adler", color="orange")
# print("t_learned_1",t_learned_1.shape)
# print("t_learned_2",t_learned_2.shape)
# print("E_learned_1", E_learned_1.shape)
# print("E_learned_2", E_learned_2.shape)
# # Set plot labels and legend
# plt.xlabel("Time")
# plt.ylabel("E")
# plt.xlim((0, 20))
# plt.ylim((0, 1.1))
# plt.legend()
# plt.show()
#Plotting the test/loss
y_1 = [1.11e-7, 9.077e-6]
x_1 = [50, 200]
y_2 = [4.64e-7, 1.23e-5]
x_2 = [50, 200]
plt.figure(figsize=(10, 6))
plt.scatter(x_1, y_1, color="black", label="Case 1")
plt.scatter(x_2, y_2, color="red", label="Case 2")
plt.xlabel('Number of Discretization')
plt.ylabel('Test/Loss')
plt.legend()
plt.show()

# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(t_1_in_reshaped, c_1_in_reshaped, label=" $c_{in}$", color="blue")
# ax1.plot(t_conv_reshaped, c_conv_reshaped, label="$c_{p,1}$", color="black", linestyle="--")
# ax1.plot(t_conv_reshaped, c_out_l_reshaped, label="$c_{o,1}$", color="yellow")
# ax1.plot(t_conv_values, c_conv_2_values, label="$c_{p,2}$", color="red", linestyle="--")
# ax1.plot(t_conv_values, c_out_values, label="$c_{o,2}$", color="orange")
# ax1.set_xlim((0, 30))
# ax1.set_ylim((0, 1.1))
# ax1.set_ylabel('Concentration')
# ax1.set_xlabel('Time')
# ax1.legend()
# plt.show()