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

# if wandb.run is not None:
#     wandb.finish()
#Parameters
tau_l = 5.0
coefficients = {
    'tau_a_val': np.array([1]),   
    'tau_p_val': np.array([2]), 
    'beta_val': np.array([0.1]), 
    'alpha_val': 0.2}
# epoch_1=30000
# epoch_2=20000
epoch_1=1
epoch_2=1
epoch_3=epoch_1
learning_rate=1e-3
disc_n_1_out=200

# base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/2_nd_Layer_Convolution'
base_dir = r'D:\Tuana\nRTD\Experiments\Preliminary\2_nd_Layer_Convolution'
n_in_1, n_out_1, n_out_2, n_e_1, n_e_2 = sp.symbols("n_in_1 n_out_1 n_out_2 n_e_1 n_e_2", positive=True, real=True)
t_1, t_adl, t_ch, t_e_1, t_e_2 = sp.symbols("t_1, t_lam, t_adl, t_e_1, t_e_2", positive=True, real=True)
equations = [
    t_1 - t_adl + t_e_1,
    n_in_1 - n_out_1 + n_e_1 - 1,
    n_in_1 / t_1 - n_e_1 / t_e_1,
    t_adl - t_ch + t_e_2,
    n_out_1 - n_out_2 + n_e_2 - 1,
    n_e_2 * t_adl - n_out_1 * t_e_2]
for equation in equations:
    print(equation)
solution_set = sp.solve(equations,(n_in_1, n_out_1, n_out_2, n_e_1, n_e_2, t_adl, t_e_2),dict=True)
print(solution_set) 
solution_set = sp.solve(
    equations,
    n_in_1,
    #n_out_1,
    n_out_2,
    n_e_1,
    n_e_2,
    # t_1,
    t_adl,
    # t_ch,
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
    n_out_1: disc_n_1_out,
    # n_e_1,
    # n_e_2,
    t_1:30,
    # t_adl,
    t_ch:120,
    t_e_1:40,
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
t_adl = result_dict.get(t_adl, sub_dict.get(t_adl))
t_ch = sub_dict.get(t_ch)
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
print("t_adl =", t_adl)
print("t_ch =", t_ch)
print("t_1 =", t_1)
print("t_e_1 =", t_e_1)
print("t_e_2 =", t_e_2)

def compute_inverse_laplace(coefficients, t_values):
    s, t = sp.symbols('s t')
    alpha = coefficients['alpha_val']
    results_out = []  
    for tau_a_val in coefficients['tau_a_val']:  
        for tau_p_val in coefficients['tau_p_val']:
            for beta_val in coefficients['beta_val']:
                tau_m_val = (beta_val * (1 - alpha)) / alpha
                F_s = (sp.exp(-tau_p_val * s)) / (1 + beta_val + tau_a_val * s - (beta_val / (1 + tau_m_val * s)))
                f_t = sp.inverse_laplace_transform(F_s, s, t)
                f_t_numeric = sp.lambdify(t, f_t, modules="numpy")
                E_t = f_t_numeric(t_values)
                results_out.append({
                    'tau_a_val': tau_a_val,
                    'tau_p_val': tau_p_val,
                    'tau_m_val': tau_m_val,
                    'beta_val': beta_val,
                    'E_t': E_t
                })
    return results_out

t_values_Adler_l = np.linspace(0, t_adl, n_1_out, endpoint=True)
results_out = compute_inverse_laplace(coefficients, t_values_Adler_l)
c_0_adl = np.zeros_like(t_values_Adler_l)
c_0_adl[t_values_Adler_l > 5] = 1

for result in results_out:
    tau_a_val = result['tau_a_val']
    tau_p_val = result['tau_p_val']
    tau_m_val = result['tau_m_val']
    beta_val = result['beta_val']
    E_Adler_out = result['E_t']
    if E_Adler_out.shape[0] != t_values_Adler_l.shape[0]:
        if E_Adler_out.shape[0] > t_values_Adler_l.shape[0]:
            E_Adler_out = E_Adler_out[:len(t_values_Adler_l)]
        else:
            E_Adler_out = np.pad(E_Adler_out, (0, len(t_values_Adler_l) - len(E_Adler_out)), 'constant')
        E_Adler_out=E_Adler_out/E_Adler_out.max()
    E_Adler_normalized_out=E_Adler_out/np.sum(E_Adler_out)
    # print(f"Shape of E_Adler_normalized: {E_Adler_normalized.shape}")

c_out_full_Adler_out = np.convolve(c_0_adl, E_Adler_normalized_out, mode="full")
t_conv_full_Adler_out = np.linspace(t_values_Adler_l[0] + t_values_Adler_l[0], t_values_Adler_l[-1] +t_values_Adler_l[-1], len(c_out_full_Adler_out))
valid_indices = t_conv_full_Adler_out <= t_adl # Adjust time limit as needed
t_conv_Adler_out = t_conv_full_Adler_out[valid_indices]
c_out_Adler_out = c_out_full_Adler_out[valid_indices]
print(c_out_Adler_out.shape,"c_out_Adler_out_inof2nd")
print(t_conv_Adler_out.shape,"t_conv_Adler_outinof2nd")

### Model Adler+Cholote, 2nd ,out
def compute_inverse_laplace(coefficients, t_values):
    s, t = sp.symbols('s t')
    alpha = coefficients['alpha_val']
    results = []  
    for tau_a_val in coefficients['tau_a_val']:  
        for tau_p_val in coefficients['tau_p_val']:
            for beta_val in coefficients['beta_val']:
                tau_m_val = (beta_val * (1 - alpha)) / alpha
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

t_values_Adler = np.linspace(0, t_ch, n_2_out, endpoint=True)
results = compute_inverse_laplace(coefficients, t_values_Adler)

for result in results:
    tau_a_val = result['tau_a_val']
    tau_p_val = result['tau_p_val']
    tau_m_val = result['tau_m_val']
    beta_val = result['beta_val']
    
    E_Adler = result['E_t']
    if E_Adler.shape[0] != t_values_Adler.shape[0]:
        if E_Adler.shape[0] > t_values_Adler.shape[0]:
            E_Adler = E_Adler[:len(t_values_Adler)]
        else:
            E_Adler = np.pad(E_Adler, (0, len(t_values_Adler) - len(E_Adler)), 'constant')
        E_Adler=E_Adler/E_Adler.max()
    E_Adler_normalized_1=E_Adler/np.sum(E_Adler)
    # print(f"Shape of E_Adler_normalized: {E_Adler_normalized.shape}")
t_values_Adler_l_c = np.linspace(0, t_ch, n_2_out, endpoint=True)
results_out = compute_inverse_laplace(coefficients, t_values_Adler_l_c)
c_1_adl = np.zeros_like(t_values_Adler_l_c)
c_1_adl[t_values_Adler_l_c > 5] = 1
c_out_full_Adler = np.convolve(c_1_adl, E_Adler_normalized_1, mode="full")
t_conv_full_Adler = np.linspace(t_values_Adler_l_c[0] + t_values_Adler_l_c[0], t_values_Adler_l_c[-1] +t_values_Adler_l_c[-1], len(c_out_full_Adler))
valid_indices = t_conv_full_Adler <= t_ch # Adjust time limit as needed
t_conv_Adler = t_conv_full_Adler[valid_indices]
c_out_Adler = c_out_full_Adler[valid_indices]

def Cholete(t: npt.NDArray[np.float64], alpha: float, beta: float, tau: float, g: float) -> npt.NDArray[np.float64]: 
    print(f"tau = {tau}")
    print(f"alpha = {alpha}")
    print(f"beta = {beta}")
    H = np.where(t < g, 0, 1)
    k = ((1 - alpha) / (beta * tau))
    exp_term = (1 - alpha) * np.exp(k * (tau - t))
    F = alpha * H - exp_term + (1 - alpha)
    F[F < 0] = 0
    E = np.gradient(F,t)  
    return F,E
def Cholete_E (t: npt.NDArray[np.float64], alpha: float, beta: float, tau: float)-> npt.NDArray[np.float64]:
    k = ((1 - alpha) / (beta * tau))
    E_c=(1-alpha)*k*np.exp(-k*t)
    return E_c
    
t = t_conv_Adler
c_0 = c_out_Adler

beta_values = np.array([0.3])
for beta in beta_values:
    F,E = Cholete(t, 0.2, beta, 5,5)
    E_c = Cholete_E(t, 0.2, beta, 5)
    #E_t_normalized = E / np.sum(E)
    E_c_normalized = E_c / np.sum(E_c)
    #print(f"beta: {beta}, Integral of E: {np.sum(E)}")
    c_out_full = np.convolve(c_0, E_c_normalized, mode="full")
    t_conv_full = np.linspace(t[0] + t[0], t[-1] + t[-1], len(c_out_full))
    valid_indices = t_conv_full <= t_ch
    t_conv_ch = t_conv_full[valid_indices]
    c_out_ch = c_out_full[valid_indices]
    
print(f"Shape of t_conv_ch_out: {t_conv_ch.shape}")
print(f"Shape of c_out_ch_out: {c_out_ch.shape}")
#print(f"Shape of E_c_normalized: {E_c_normalized.shape}")
print(f"Shape of c_out_Adler_outsupp: {c_out_Adler.shape}")
print(f"Shape of t_conv_Adleroutsupp for discretization: {t_conv_Adler.shape}")
#print(f"Shape of E_Adler_normalized for discretization: {E_Adler_normalized.shape}")

#NN
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True,name=f'AdCh_Layer_1_n_1_out_{disc_n_1_out}_LC_learning_rate_{learning_rate}_1st_epoch_{epoch_1}',
    reinit=True
)
c_conv_1_results = {}
t_1_in = torch.linspace(0, t_1, n_1_in).unsqueeze(0).unsqueeze(0).float()
c_1_in = torch.zeros((1, 1, n_1_in)).float()
indices = (t_1_in > 5).squeeze()
c_1_in[:, :, indices] = 1
print("c_1_in shape:", c_1_in.shape)  
print("t_1_in shape:", t_1_in.shape) 
c_out_adl = torch.tensor(c_out_Adler_out).unsqueeze(0).unsqueeze(0)  
t_conv_adl = torch.tensor(t_conv_Adler_out).unsqueeze(0).unsqueeze(0) 
c_out = torch.cat([torch.tensor(c_out_adl)], dim=0).float()
t_conv = torch.cat([torch.tensor(t_conv_adl)], dim=0).float()
c_out_list = [c_out_adl]
t_conv_list = [t_conv_adl]
c_out = torch.cat(c_out_list, dim=-1)
t_conv = torch.cat(t_conv_list, dim=-1)
 
model = RTDModule(
    kernel_sizes=[n_e_1],
    kernel_times=[(0.0, t_e_1)],
    learning_rate=learning_rate,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},)

c_conv = model(c_1_in)
print("c_1_in",c_1_in.shape)
print("c_conv",c_conv.shape)
E = model.net.E[0]
t_E = torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
E /= E.max()
print(model(c_1_in).size())
ds = TensorDataset(c_1_in.float(), c_out_adl.float())
print("c_1_in.float()",c_1_in.float().shape)
print("c_out_l.float()",c_out_adl.float().shape)
dl = DataLoader(ds, batch_size=1, shuffle=True)
trainer = pl.Trainer(accelerator="auto", max_epochs=epoch_1, logger=wandb_logger,deterministic=True)
# trainer = pl.Trainer(accelerator="auto", max_epochs=epoch,deterministic=True)
trainer.fit(model, dl)
trainer.test(model, dl)
c_conv = model(c_1_in)
c_conv_1_results[n_1_out] = c_conv.detach().numpy()
print("c_conv_results",c_conv.detach().numpy().shape)
E = model.net.E[0]
t_E = torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
E = E / E.max()

# Plotting
fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)
ax1.plot(t_1_in.squeeze().numpy(), c_1_in[0, 0, :].squeeze().numpy(), label="Input Signal", color="blue")
ax1.plot(t_conv[0, 0, :].squeeze().numpy(), c_out_adl.squeeze().numpy(), label="Expected Output", color="green")
ax1.plot(t_conv[0, 0, :].squeeze().numpy(), c_conv[0, 0, :].detach().squeeze().numpy(), label="Predicted Output", color="red", linestyle="--")
ax1.plot(t_E, E, label="Convolution Kernel", color="purple", linestyle="--")
# ax1.plot(t_conv_l, E_laminar, color="grey")
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.set_xlabel('Time')
ax1.legend()
plt.show()
wandb.finish()

###############################################################################
##Second Layer
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True,name=f'Layer_2_CH_n_1_out_{disc_n_1_out}_LC_learning_rate_{learning_rate}_2nd_epoch_{epoch_2}',
    reinit=True
)
c_conv_2_results = {}
t_2_in =torch.tensor(t_conv).float()
c_2_in = torch.tensor(c_conv).float()
c_out_ch=torch.tensor(c_out_ch).float().unsqueeze(0).unsqueeze(0)
print("c_2_in",c_2_in.shape)
print("Shape of c_out_ch:", c_out_ch.shape) 
t_out_ch = torch.tensor(t_conv_ch).float().unsqueeze(0).unsqueeze(0)



print(t_2_in.shape)
print("c_2_in",c_2_in.shape)
print(c_out_ch.shape)
print(t_out_ch.shape)

model_2 = RTDModule(
    kernel_sizes=[n_e_2],
    kernel_times=[(0.0, t_e_2)],
    learning_rate=learning_rate,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

c_conv_2 = model_2(c_2_in)
print(c_conv_2.shape)
E_2 = model_2.net.E[0]  
t_E_2 = torch.linspace(0, t_e_2, model_2.kernel_sizes[0])
E_2 = E_2 / E_2.max()
print(E_2.shape)
print(t_E_2.shape)


# Create TensorDataset with consistent tensor types
ds_2 = TensorDataset(c_2_in.float(), c_out_ch.float())
dl_2 = DataLoader(ds_2, batch_size=1, shuffle=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=epoch_2,
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
t_E_2 = torch.linspace(0, t_e_2, model_2.kernel_sizes[0])
E_2 = E_2 / E_2.max()
c_conv_2 = model_2(c_2_in)
c_conv_2_results[n_2_out] = c_conv_2.detach().numpy()  
fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(10, 8))
ax1.plot(t_2_in.squeeze().numpy(), c_2_in.squeeze().numpy(), label="Input Signal", color="blue", linestyle="--")
ax1.plot(t_out_ch.squeeze().numpy(), c_out_ch.squeeze().numpy(), label="Expected Output", color="green")
ax1.plot(t_out_ch.squeeze().numpy(), c_conv_2.detach().squeeze().numpy(), label="Predicted Output", color="red", linestyle="--")
ax1.plot(t_E_2, E_2, label="Convolution Kernel", color="purple", linestyle="--")
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.set_xlabel('Time')
ax1.legend()
plt.show()
wandb.finish()

###Plotting
t_1_in_reshaped = t_1_in.squeeze().numpy()
c_1_in_reshaped = c_1_in[0, 0, :].squeeze().numpy()
t_conv_reshaped = t_conv[0, 0, :].squeeze().numpy()
c_conv_reshaped = c_conv[0, 0, :].detach().squeeze().numpy()

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(t_1_in_reshaped, c_1_in_reshaped, label=" $c_{in}$", color="blue")
ax1.plot(t_conv_reshaped, c_conv_reshaped, label="$c_{p,1}$", color="black", linestyle="--")
ax1.plot(t_conv[0, 0, :].squeeze().numpy(), c_out_adl.squeeze().numpy(), label="$c_{o,1}$", color="yellow")
ax1.plot(t_out_ch.squeeze().numpy(), c_out_ch.squeeze().numpy(), label="$c_{o,2}$", color="orange")
ax1.plot(t_out_ch.squeeze().numpy(), c_conv_2.detach().squeeze().numpy(), label="$c_{p,2}$", color="red", linestyle="--")
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.set_xlabel('Time')
ax1.legend()
plt.show()


import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots.figures import Elsevier_Sizes
import datetime
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots.figures import Elsevier_Sizes, ACS_Sizes
from ICIW_Plots import make_square_ax, cm2inch
from ICIW_Plots import make_rect_ax
plt.style.use("ICIWstyle")
fig = plt.figure(figsize=(Elsevier_Sizes.single_column["in"], 12 * cm2inch))
ax = make_rect_ax(
    fig,
    ax_width=7.3 * cm2inch,
    ax_height=5 * cm2inch,
    # left_h=0.2,  # These arguments control the spacing of the axis
    # bottom_v=0.2, # not supplying them wil place the axes in the middle of the figure
    xlabel=r"$t$ / $s$",
    ylabel=r"$E$ / $1$"
)
ax.plot(t_1_in_reshaped, c_1_in_reshaped, label=r"$x_{0(t)}$",color=ICIWcolors.CERULEAN)
ax.plot(t_conv_reshaped, c_conv_reshaped, label="$\hat{x}_{1(t)}$",color="purple",linestyle="--")
ax.plot(t_conv[0, 0, :].squeeze().numpy(), c_out_adl.squeeze().numpy(), label=r"$x_{1(t)}$",color=ICIWcolors.DRAB)
ax.plot(t_out_ch.squeeze().numpy(), c_conv_2.detach().squeeze().numpy(), label="$\hat{x}_{2(t)}$", color="black", linestyle="--")
ax.plot(t_out_ch.squeeze().numpy(), c_conv_2.detach().squeeze().numpy(), label=r"$x_{2(t)}$", color=ICIWcolors.FLAME)
ax.legend(loc='best')
ax.set_xlim((0, 30))
ax.set_ylim((-0.1, 1.1))
current_date = datetime.datetime.now().strftime("%Y%m%d")
plt.savefig(f"2nd_Layer_Ch_Adl_{current_date}.png", dpi=300)
plt.show()


E_Adler_normalized = E_Adler / np.max(E_Adler)
# E_learned_1 = model.net.E[0] if isinstance(model.net.E[0], np.ndarray) else model.net.E[0].numpy()
# #E_learned_2 = model_2.net.E[0] if isinstance(model_2.net.E[0], np.ndarray) else model_2.net.E[0].numpy()
# E_learned_1 /= np.max(E_learned_1)
#E_learned_2 /= np.max(E_learned_2)


#t_adler = t_values_Adler if isinstance(t_values_Adler, np.ndarray) else t_values_Adler.numpy()
t_learned_1 = np.linspace(0, t_e, len(E))
t_learned_2 = np.linspace(0, t_e_2, len(E_2))


plt.figure(figsize=(10, 6))
# plt.plot(t_l, E_laminar, label="E Laminar Layer 1", color="blue")
#plt.plot(t_l_a, E_laminar_a_normalized, label="E Laminar Layer 2", color="green")
# plt.plot(t_adler, E_Adler_normalized, label="E Adler", color="orange")
plt.plot(t_learned_1, E, label="$E_1$", color="black")
plt.plot(t_learned_2, E_2, label="$E_2$", color="red")
print("t_learned_1",t_learned_1.shape)
print("t_learned_2",t_learned_2.shape)
#print("E_learned_1", E_learned_1.shape)
#print("E_learned_2", E_learned_2.shape)
# Set plot labels and legend
plt.xlabel("Time")
plt.ylabel("E")
plt.xlim((0, 20))
plt.ylim((0, 1.1))
plt.legend()
plt.show()

plt.style.use("ICIWstyle")
fig = plt.figure(figsize=(Elsevier_Sizes.single_column["in"], 12 * cm2inch))
ax = make_rect_ax(
    fig,
    ax_width=7.3 * cm2inch,
    ax_height=5 * cm2inch,
    # left_h=0.2,  # These arguments control the spacing of the axis
    # bottom_v=0.2, # not supplying them wil place the axes in the middle of the figure
    xlabel=r"$t$ / $s$",
    ylabel=r"$E$ / $1$"
)
ax.plot(t_learned_1, E, label="$\hat{E}_{1(t)}$", color="purple")
ax.plot(t_learned_2, E_2, label="$\hat{E}_{2(t)}$", color=ICIWcolors.FLAME)
ax.legend(loc='best')
ax.set_xlim((0, 30))
ax.set_ylim((-0.1, 1.1))
current_date = datetime.datetime.now().strftime("%Y%m%d")
plt.savefig(f"2nd_Layer_Adl_Lam_E{current_date}.png", dpi=300)
plt.show()

# #Plotting the test/loss
# y_1 = [1.11e-7, 9.077e-6]
# x_1 = [50, 200]
# y_2 = [4.64e-7, 1.23e-5]
# x_2 = [50, 200]

# plt.figure(figsize=(10, 6))
# plt.scatter(x_1, y_1, color="black", label="Case 1")
# plt.scatter(x_2, y_2, color="red", label="Case 2")

# plt.xlabel('Number of Discretization')
# plt.ylabel('Test/Loss')
# plt.legend()
# plt.show()

