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
from nRTD.rtd_fitting_2 import RTDModule
from nRTD.rtd_net_4 import RTDNet
from lightning.pytorch import loggers as pl_loggers
import os
import datetime
import sympy as sp
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch

if wandb.run is not None:
    wandb.finish()
# module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
# sys.path.append(module_path)

#laminar_model_dir='/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model'
laminar_model_dir=r"D:\Tuana\nRTD\Experiments\Preliminary\Convolution_script\Convolution_021_Laminar_Flow_Model\Convolution_021_Laminar_Flow_Model.py"
tau_5_dir = r"D:\Tuana\nRTD\Experiments\Preliminary\Litrature\Laminar_Flow_Model\tau_5.0_disc_200"

#tau_5_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model/tau_5.0_disc_200'

epoch=17000
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
    n_out_1: 200,
    # n_e_1,
    t_i: 30,
    t_o:60,
    t_e_1:30,
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

###CNN
t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
n_disc = n_in_1
t_input = torch.linspace(0, t_i, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 5] = 1
c_in[1::2, :, t_input < 5] = 1
c_out_list = []
t_conv_list = []
c_out_list.append(c_out_tau)
t_conv_list.append(t_conv_tau)
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

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
print("Kernel sizes:", model.kernel_sizes)
t_E =torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
c_conv = model(c_in)
E = model.net.E[0]
E = E / E.max()
j = 0  
plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")
for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label="Tau 5.0",
        color="green")
plt.plot(
    t_conv[j, 0, :].numpy(),
    c_conv[j, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.legend()
plt.xlim((0, 40))
plt.ylim((0, 1.1))
plt.show()
print(model(c_in).size())
ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)

wandb.init()
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True
)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=epoch,
    logger=wandb_logger,
    deterministic=True,
)

# trainer = pl.Trainer(
#     accelerator="auto",
#     max_epochs=epoch,
#     deterministic=True,
# )

trainer.fit(model, dl)
trainer.test(model, dl)
c_conv = model(c_in)
E = model.net.E[0]
t_E =torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
E = E / E.max()
t_E_np = np.linspace(0, 30, 100)

def expected_formula(t):
    return np.where(t >= 2.5, (5**2) / (2 * (t**3)), 0)
E_expected_np = expected_formula(t_E_np)
E_expected_np = E_expected_np / E_expected_np.max()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)  
ax1.plot(t_input, c_in[0, 0, :].numpy(), label="$SF$", color="blue")
ax1.plot(t_conv[0, 0, :].numpy(), c_conv[0, 0, :].detach().numpy(), color="red",label=r'$x_{CNN,lam}$')
for i in range(c_out.size(1)):
    ax1.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label=r'$x_{th,lam}$',
        color="black", linestyle="--", linewidth=2.2
    )
ax1.set_xlim((0, 31))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('$x$ / $1$')
ax1.legend()
ax1.tick_params(labelbottom=False)
ax2.plot(t_E, E, label="$E_{CNN,lam}$",  color="red")
ax2.set_xlabel('$t$ / $s$')
ax2.set_ylabel('$E$ / $1$')
ax2.plot(t_E_np, E_expected_np, label='$E_{th,lam}$', color='black',linestyle="--", linewidth=2.2)
ax2.set_xlim((0, 31))
ax2.set_ylim((0, 1.1))
ax2.legend()
plt.xlabel("$t$ / $s$")
current_date = datetime.datetime.now().strftime("%Y%m%d")
plt.savefig(f"Figure_plot_{current_date}.png", dpi=300)
plt.show()

#Saving of E and time
predicted_E = E
predicted_time = t_E.numpy()               
expected_E = E_expected_np                   
expected_time = t_E_np
c_conv_in_50=c_conv.detach().numpy()
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model"
np.save(os.path.join(save_dir, 'E_predicted.npy'), predicted_E)
np.save(os.path.join(save_dir, 't_E_predicted.npy'), predicted_time)
np.save(os.path.join(save_dir, 'E_expected.npy'), expected_E)
np.save(os.path.join(save_dir, 't_E_expected.npy'), expected_time)
np.save(os.path.join(save_dir, 'c_conv_in_100.npy'),c_conv_in_50 )
print("saved under:", save_dir)
predicted_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/E_predicted.npy')
predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/t_E_predicted.npy')
plt.plot(predicted_time, predicted_E, label='$E_{CNN}$', color='orange')
plt.plot(expected_time, expected_E, label='$E_{th,lam}$', color='purple', linestyle='--')
plt.xlabel('$t$ / $s$')
plt.ylabel('$E$ / $1$')
plt.xlim((predicted_time.min(), predicted_time.max()))
plt.xlim((0, 30))  
plt.ylim((0, 1.1))  
plt.legend()
current_date = datetime.datetime.now().strftime("%Y%m%d")
plt.savefig(f"E_saved_{current_date}.png", dpi=300)
plt.show()

print(f"c_conv_in size: {t_E.numpy().shape}")
print(f"c_conv_in_100 shape: {c_conv_in_50.shape}")

import torch
import matplotlib.pyplot as plt
import numpy as np
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots.figures import Elsevier_Sizes
import datetime
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch


plt.style.use("ICIWstyle")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(Elsevier_Sizes.double_column["in"], 12 * cm2inch))
ax1.plot(t_input.numpy(), c_in[0, 0, :].numpy(), label=r"$x_{0(t)}$", color=ICIWcolors.CERULEAN)
for i in range(c_out.size(1)):
    ax1.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label=r"$x_{(t)}$",
       color=ICIWcolors.DRAB
    )
ax1.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label=r"$\hat{x}_{(t)}$",
    color="purple",
    linestyle="--"
)
ax1.set_ylabel(r"$x$ / $1$", )
ax1.legend(loc='best')
ax1.set_xlim((0, 30))  
ax1.set_ylim((-0.1, 1.1))  
ax2.plot(expected_time, expected_E, label=r"$E_{(t)}$", color=ICIWcolors.KELLYGREEN)
ax2.plot(predicted_time, predicted_E, label=r"$\hat{E}_{(t)}$", color="black", linestyle="--")
ax2.set_xlabel(r"$t$ / $s$")
ax2.set_ylabel(r"$E$ / $1$")
ax2.legend(loc='best')
ax2.set_xlim((0, 30))  
ax2.set_ylim((-0.1, 1.1)) 
current_date = datetime.datetime.now().strftime("%Y%m%d")
plt.savefig(f"Profile_Laminar_{current_date}.png", dpi=300)
plt.show()

