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
# module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
# sys.path.append(module_path)


adler_dir = r'D:\Tuana\nRTD\Experiments/Preliminary/Litrature/Adler_havarka_Model/tau_a_val_3_tau_p_val_2_tau_m_val_0.4000000000000001_beta_val_0.1_2210'
#adler_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Adler_havarka_Model/tau_a_val_3_tau_p_val_2_tau_m_val_0.4000000000000001_beta_val_0.1_2210'
epoch=26000
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
    t_i: 40,
    t_o:70,
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

t_conv_tau = torch.tensor(np.load(os.path.join(adler_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(adler_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
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
    learning_rate=1e-4,
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

wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True,name="tau3_adl"
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

def compute_inverse_laplace(coefficients, t_values):
    s, t = sp.symbols('s t', real=True, positive=True)
    alpha = coefficients['alpha_val']
    results = []  
    for tau_a_val in coefficients['tau_a_val']:
        for tau_p_val in coefficients['tau_p_val']:
            for beta_val in coefficients['beta_val']:
                tau_m_val = (beta_val * (1 - alpha)) / alpha
                # Laplace transform equation
                F_s = (sp.exp(-tau_p_val * s)) / (1 + beta_val + tau_a_val * s - (beta_val / (1 + tau_m_val * s)))
                f_t = sp.inverse_laplace_transform(F_s, s, t)
                f_t_numeric = sp.lambdify(t, f_t, modules="numpy")
                E_t = f_t_numeric(t_values)
                results.append({
                    'tau_a_val': tau_a_val,
                    'tau_p_val': tau_p_val,
                    'tau_m_val': tau_m_val,
                    'beta_val': beta_val,
                    'E_t': E_t
                })
    return results
coefficients = {
    'tau_a_val': np.array([3]),
    'tau_p_val': np.array([2]), #####check!
    'beta_val': np.array([0.1]),
    'alpha_val': 0.2
}
t_plot = np.linspace(0, t_e_1,n_e_1)
inverse_laplace_results = compute_inverse_laplace(coefficients, t_plot)
E_expected = inverse_laplace_results[0]['E_t']
E_expected =E_expected /E_expected.max()

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
ax2.plot(t_plot, E_expected, label='$E_{th,lam}$', color='black',linestyle="--", linewidth=2.2)
ax2.set_xlim((0, 31))
ax2.set_ylim((0, 1.1))
ax2.legend()
plt.xlabel("$t$ / $s$")
current_date = datetime.datetime.now().strftime("%Y%m%d")
plt.savefig(f"Figure_plot_{current_date}.png", dpi=300)
plt.show()


predicted_E = E
predicted_time = t_E              
expected_E = E_expected                 
expected_time = t_plot
#save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_023_Adler_havarka_Model"
save_dir=r'D:\Tuana\nRTD\Experiments\Preliminary\Convolution_script\Convolution_023_Adler_havarka_Model'
np.save(os.path.join(save_dir, 'E_predicted_tau_a_val_3.npy'), predicted_E)
np.save(os.path.join(save_dir, 't_E_predicted_tau_a_val_3.npy'), predicted_time)
np.save(os.path.join(save_dir, 'E_expected_tau_a_val_3.npy'), expected_E)
np.save(os.path.join(save_dir, 't_E_expected_tau_a_val_3.npy'), expected_time)

predicted_E = np.load(r'D:\Tuana\nRTD\Experiments\Preliminary\Convolution_script\Convolution_023_Adler_havarka_Model\E_predicted_tau_a_val_3.npy')
predicted_time = np.load(r'D:\Tuana\nRTD\Experiments\Preliminary\Convolution_script\Convolution_023_Adler_havarka_Model\t_E_predicted_tau_a_val_3.npy')
plt.plot(t_E, E, label="E", color="orange")
plt.plot(expected_time-1, expected_E, label='E_expected_tau_a_val_1', color='purple', linestyle='--')
plt.xlabel('t')
plt.ylabel('E')
plt.xlim((predicted_time.min(), predicted_time.max()))
plt.ylim((0, 1.1))  
plt.xlim(0,25)
plt.legend()
plt.xticks(np.arange(0, 10, 1))  
#plt.savefig(os.path.join(unified_dir, 'E_saved_21102024_tau_a_val_1.png'), dpi=300)
plt.show()


import torch
import matplotlib.pyplot as plt
import numpy as np
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots.figures import Elsevier_Sizes
import datetime
from ICIW_Plots import cm2inch

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
ax1.set_xlim((0, 20))  
ax1.set_ylim((-0.1, 1.1))  
ax2.plot(t_E, E_expected, label=r"$E_{(t)}$", color=ICIWcolors.KELLYGREEN)
ax2.plot(t_E, E, label=r"$\hat{E}_{(t)}$", color="black", linestyle="--")
ax2.set_xlabel(r"$t$ / $s$")
ax2.set_ylabel(r"$E$ / $1$")
ax2.legend(loc='best')
ax2.set_xlim((0, 20))  
ax2.set_ylim((-0.1, 1.1)) 
current_date = datetime.datetime.now().strftime("%Y%m%d")
plt.savefig(f"Profiles_tau_3_{current_date}.png", dpi=300)
plt.show()



