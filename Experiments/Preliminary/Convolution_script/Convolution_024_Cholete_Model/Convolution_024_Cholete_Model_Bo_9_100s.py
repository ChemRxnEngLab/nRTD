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
module_path = r"D:\Tuana\nRTD\lib"
sys.path.append(module_path)
#module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
#sys.path.append(module_path)
from nRTD.rtd_fitting_2 import RTDModule
from nRTD.rtd_net_4 import RTDNet
from lightning.pytorch import loggers as pl_loggers
import os
import datetime
import sympy as sp
from sympy import ceiling
import matplotlib.pyplot as plt
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots.figures import Elsevier_Sizes, ACS_Sizes
from ICIW_Plots import make_square_ax, cm2inch
from ICIW_Plots import make_rect_ax
import datetime

if wandb.run is not None:
    wandb.finish()

#Bo_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Cholete_Model/beta0.9_100sc'
Bo_dir = r'D:\Tuana\nRTD\Experiments\Preliminary\Litrature\Cholete_Model\beta0.9_100sc'
t_conv_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

epoch=16000
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
    t_i: 50,
    t_o:100,
    t_e_1:50,
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

n_disc = n_in_1
t_input = torch.linspace(0, t_i, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[:, :, t_input > 5] = 1
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
    learning_rate=1e-2,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)
print("Kernel sizes:", model.kernel_sizes)
t_E =torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
c_conv = model(c_in)
E = model.net.E[0]
E = E / E.max()
j = 0  

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

def Cholete(t: npt.NDArray[np.float64], alpha: float, beta: float, tau: float, g: float) -> npt.NDArray[np.float64]: 
    print(f"tau = {tau}")
    print(f"alpha = {alpha}")
    print(f"beta = {beta}")
    H = np.where(t < g, 0, 1)
    k = ((1 - alpha) / (beta * tau))
    exp_term = (1 - alpha) * np.exp(k * (tau - t))
    F = alpha * H - exp_term + (1 - alpha)
    F[F < 0] = 0
    E_e = np.gradient(F,t)  
    return F,E_e

def Cholete_E (t: npt.NDArray[np.float64], alpha: float, beta: float, tau: float)-> npt.NDArray[np.float64]:
    k = ((1 - alpha) / (beta * tau))
    E_c_e=(1-alpha)*k*np.exp(-k*t)
    return E_c_e
    
t_exp = np.linspace(0, t_e_1, n_e_1)
c_0_exp = np.zeros_like(t_exp)
c_0_exp[t_exp > 5] = 1
beta_values = np.array([0.9])
for beta in beta_values:
    F,E_e = Cholete(t_exp, 0.2, beta, 5,5)
    E_c_e = Cholete_E(t_exp, 0.2, beta, 5)
    #E_t_normalized = E / np.sum(E)
    E_c_e_normalized = E_c_e / np.sum(E_c_e)
    #print(f"beta: {beta}, Integral of E: {np.sum(E)}")
    c_out_full = np.convolve(c_0_exp, E_c_e_normalized, mode="full")
    t_conv_full = np.linspace(t_exp[0] + t_exp[0], t_exp[-1] + t_exp[-1], len(c_out_full))
    valid_indices = t_conv_full <= 50
    t_conv_e = t_conv_full[valid_indices]
    c_out_e = c_out_full[valid_indices]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)  
ax1.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    ax1.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="Bo=0.9",
        color="green",
    )


ax1.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",linestyle="-."
)
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.set_xticks(np.arange(0, 7, 1)) 
ax1.legend()
ax1.tick_params(labelbottom=False)
ax2.plot(t_E, E, label="E", color="orange")
ax2.plot(t_E, E_c_e_normalized/E_c_e_normalized.max(), label="E", color="red",linestyle="--")
#ax2.plot(t_values, E_expected, label="E (Expected )", color="purple", linestyle="--")
ax2.set_xlabel('t')
ax2.set_ylabel('E')
#ax2.plot(t_values, E_expected, label="E (Expected )", color="purple", linestyle="--")
#ax2.plot(t_conv, E, label="E (Expected )", color="purple", linestyle="--")
ax2.set_xlim((0, 50))
ax2.set_ylim((0, 1.1))
ax2.legend()
ax2.set_xticks(np.arange(0, 10, 1))  
plt.xlabel("Time")
save_dir = "D:\Tuana\nRTD\Experiments\Preliminary\Convolution_script\Convolution_024_Cholete_Model\Convolution_024_Cholete_Model_B_01.py"
# unified_dir = os.path.join(save_dir, f'Bo_{0.9}')
# os.makedirs(unified_dir, exist_ok=True)
# plt.savefig(os.path.join(unified_dir, 'Figure_Cholete_09.png'), dpi=300)
plt.show()
print("E",E.shape)
print("E_c_e_normalized",E_c_e_normalized.shape)

predicted_E = E
predicted_time = t_E.numpy()               
expected_E = E_c_e_normalized                   
expected_time = t_conv_e
#c_conv=c_conv.detach().numpy()
c_conv=c_conv.detach().numpy()
save_dir =r"D:\Tuana\nRTD\Experiments\Preliminary\Convolution_script\Convolution_024_Cholete_Model"
np.save(os.path.join(save_dir, 'E_predicted_09_100s.npy'), predicted_E)
np.save(os.path.join(save_dir, 't_E_predicted_09_100s.npy'), predicted_time)
np.save(os.path.join(save_dir, 'E_expected_09_100s.npy'), expected_E)
np.save(os.path.join(save_dir, 't_E_expected_09_100s.npy'), expected_time)
np.save(os.path.join(save_dir, 'c_conv_in_09.npy'),c_conv )
print("saved under:", save_dir)

#####

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
ax1.plot(t_input.numpy(), c_in[0, 0, :].numpy(), label=r"$C_0(t)$", color=ICIWcolors.CERULEAN)
for i in range(c_out.size(1)):
    ax1.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label=r"$C(t)$",
       color=ICIWcolors.DRAB
    )
ax1.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :],
    label=r"$\hat{C}(t)$",
    color="purple",
    linestyle="--"
)
ax1.set_ylabel(r"$C$ / $1$", )
ax1.legend(loc='best')
ax1.set_xlim((0, 30))  
ax1.set_ylim((-0.1, 1.1))  
ax2.plot(t_E, E_c_e_normalized/E_c_e_normalized.max(), label=r"$E(t)$", color=ICIWcolors.KELLYGREEN)
ax2.plot(t_E, E, label=r"$\hat{E}(t)$", color="black", linestyle="--")
ax2.set_xlabel(r"$t$ / $s$")
ax2.set_ylabel(r"$E$ / $1$")
ax2.legend(loc='best')
ax2.set_xlim((0, 30))  
ax2.set_ylim((-0.1, 1.1)) 
current_date = datetime.datetime.now().strftime("%Y%m%d")
plt.savefig(os.path.join(save_dir,f"Profile_Ch_09_100s_{current_date}.png"), dpi=300)
plt.show()