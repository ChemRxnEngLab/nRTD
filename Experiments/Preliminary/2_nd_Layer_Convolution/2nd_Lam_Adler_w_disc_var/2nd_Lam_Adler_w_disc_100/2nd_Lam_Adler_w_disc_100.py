import matplotlib.pyplot as plt
import numpy.typing as npt
import sys
import os
module_path = r"D:\Tuana\nRTD\lib"
sys.path.append(module_path)
# module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
# sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import numpy as np
import wandb
from nRTD import RTDModule
from lightning.pytorch import loggers as pl_loggers
import os
from datetime import datetime
import sympy as sp
from sympy import ceiling


if wandb.run is not None:
    wandb.finish()
    
#Parameters
tau_l = 5.0
coefficients = {
    'tau_a_val': np.array([1]),   
    'tau_p_val': np.array([2]), 
    'beta_val': np.array([0.1]), 
    'alpha_val': 0.2}

epoch_1=15000
epoch_2=450000
epoch_3=epoch_1
learning_rate=1e-3
disc_n_1_out=100

# base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/2_nd_Layer_Convolution'
n_in_1, n_out_1, n_out_2, n_e_1, n_e_2 = sp.symbols("n_in_1 n_out_1 n_out_2 n_e_1 n_e_2", positive=True, real=True)
t_1, t_lam, t_adl, t_e_1, t_e_2 = sp.symbols("t_1, t_lam, t_adl, t_e_1, t_e_2", positive=True, real=True)
equations = [
    t_1 - t_lam + t_e_1,
    n_in_1 - n_out_1 + n_e_1 + 1,
    n_in_1 / t_1 - n_e_1 / t_e_1,
    t_lam - t_adl + t_e_2,
    n_out_1 - n_out_2 + n_e_2 + 1,
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
    n_out_1: disc_n_1_out,
    # n_e_1,
    # n_e_2,
    t_1:30,
    # t_lam,
    t_adl:120,
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


### Model Laminar### , For the first CNN as output
def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E_laminar = np.zeros_like(t)
    E_laminar[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E_laminar
t_l = np.linspace(0, t_lam, n_1_out, endpoint=True)  
c_0_l = np.zeros_like(t_l)
c_0_l[t_l > 5] = 1  
E_laminar = laminarflow(t_l, tau_l)
E_laminar = E_laminar / E_laminar.max()
c_out_l_full = np.convolve(c_0_l, E_laminar / np.sum(E_laminar), mode="full")
t_conv_l_full = np.linspace(t_l[0] + t_l[0], t_l[-1] + t_l[-1], len(c_out_l_full))
valid_indices = t_conv_l_full <= t_lam
t_conv_l = t_conv_l_full[valid_indices]
c_out_l = c_out_l_full[valid_indices]
print(f"Shape of c_out_l: {c_out_l.shape}")

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
# fig.subplots_adjust(hspace=0)
# ax1.plot(t_l, E_laminar, label=f'Tau {tau_l}')
# ax1.set_xlim(0,5)
# ax1.set_xlabel('t')
# ax1.set_ylabel('E')
# #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #plt.savefig(f"Laminar_Model_{timestamp}_n_1_out_{disc_n_1_out}.png", format="png", dpi=300)
# ax1.legend()

### Model Laminar### , For the middle(required for the second CNN as expected out) CNN as output
def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E_laminar_3 = np.zeros_like(t)
    E_laminar_3[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E_laminar_3
t_l_3 = np.linspace(0, t_adl, n_2_out, endpoint=True)  
c_0_l_3 = np.zeros_like(t_l_3)
c_0_l_3[t_l_3 > 5] = 1  
E_laminar_3 = laminarflow(t_l_3, tau_l)
E_laminar_3 = E_laminar_3 / E_laminar_3.max()
c_out_l_3_full = np.convolve(c_0_l_3, E_laminar_3 / np.sum(E_laminar_3), mode="full")
t_conv_l_3_full = np.linspace(t_l_3[0] + t_l_3[0], t_l_3[-1] + t_l_3[-1], len(c_out_l_3_full))
valid_indices = t_conv_l_3_full <= t_adl
t_conv_l_3 = t_conv_l_3_full[valid_indices]
c_out_l_3 = c_out_l_3_full[valid_indices]
print(f"Shape of c_out_l: {c_out_l_3.shape}")


### Adler Model###
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

t_values_Adler = np.linspace(0, t_adl, n_2_out, endpoint=True)
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
    E_Adler_normalized=E_Adler/np.sum(E_Adler)
    # print(f"Shape of E_Adler_normalized: {E_Adler_normalized.shape}")

def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E_laminar_a = np.zeros_like(t)
    E_laminar_a[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E_laminar_a
t_l_a = np.linspace(0, t_adl, n_2_out, endpoint=True)  
c_0_l_a = np.zeros_like(t_l_a)
c_0_l_a[t_l_a > 5] = 1  
E_laminar_a = laminarflow(t_l_a, tau_l)
E_laminar_a = E_laminar_a / E_laminar_a.max()
c_out_l_a_full = np.convolve(c_0_l_a, E_laminar_a / np.sum(E_laminar_a), mode="full")
t_conv_l_a_full = np.linspace(t_l_a[0] + t_l_a[0], t_l_a[-1] + t_l_a[-1], len(c_out_l_a_full))
valid_indices = t_conv_l_a_full <= t_adl  
t_conv_l_a = t_conv_l_a_full[valid_indices]  
c_out_l_a = c_out_l_a_full[valid_indices] 
c_out_l_a = c_out_l_a.flatten() 

c_out_full_Adler = np.convolve(c_out_l_a, E_Adler_normalized, mode="full")
t_conv_full_Adler = np.linspace(t_values_Adler[0] + t_values_Adler[0], t_values_Adler[-1] +t_values_Adler[-1], len(c_out_full_Adler))
valid_indices = t_conv_full_Adler <= t_adl # Adjust time limit as needed
t_conv_Adler = t_conv_full_Adler[valid_indices]
c_out_Adler = c_out_full_Adler[valid_indices]

plt.figure(figsize=(12, 6))
plt.plot(t_conv_l_a, c_out_l_a, label="Input_Laminar_NN", color="blue")
plt.plot(t_conv_Adler, c_out_Adler, label="Output from the model", color="orange")
plt.xlabel("Time", fontsize=14)
plt.ylabel("conc", fontsize=14)
plt.legend(fontsize=12)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"Adler_model_and_Laminar_model_theory_{timestamp}_n_1_out{disc_n_1_out}.png", format="png", dpi=300)
plt.show()
# print(f"Shape of t_conv_Adler: {t_conv_Adler.shape}")
# print(f"Shape of c_out_Adler: {c_out_Adler.shape}")
# print(f"Shape of t_values_Adler: {t_values_Adler.shape}")
# print(f"Shape of E_Adler_normalized: {E_Adler_normalized.shape}")
# print(f"Shape of c_out_l_a for discretization: {c_out_l_a.shape}")
# print(f"Shape of t_conv_l_a for discretization: {t_conv_l_a.shape}")
###############################################################################

#NN
c_conv_results = {}
t_1_in = torch.linspace(0, t_1, n_1_in).unsqueeze(0).unsqueeze(0).float()
c_1_in = torch.zeros((1, 1, n_1_in)).float()
indices = (t_1_in > 5).squeeze()
c_1_in[:, :, indices] = 1
print("c_1_in shape:", c_1_in.shape)  
print("t_1_in shape:", t_1_in.shape) 
c_out_l = torch.tensor(c_out_l).unsqueeze(0).unsqueeze(0)  
t_conv_l = torch.tensor(t_conv_l).unsqueeze(0).unsqueeze(0) 
c_out = torch.cat([torch.tensor(c_out_l)], dim=0).float()
t_conv = torch.cat([torch.tensor(t_conv_l)], dim=0).float()
c_out_list = [c_out_l]
t_conv_list = [t_conv_l]
c_out = torch.cat(c_out_list, dim=-1)
t_conv = torch.cat(t_conv_list, dim=-1)
print("c_out shape:", c_out.shape)  
print("t_conv shape:", t_conv.shape) 
 
model = RTDModule(
        kernel_size=n_e_1,
        learning_rate=learning_rate,
        use_scheduler=True,
        scheduler_kwargs={"factor": 0.5, "patience": 80},
    )
wandb_logger = pl_loggers.WandbLogger(
project="nRTD",
log_model=True,name=f'Model_1_n_1_out_{disc_n_1_out}_learning_rate_{learning_rate}_1st_epoch_{epoch_1}',
reinit=True
)

c_conv = model(c_1_in)
print("c_1_in",c_1_in.shape)
print("c_conv",c_conv.shape)
E = model.net.E[0]
t_E = torch.linspace(0, float(t_e_1), int(model.kernel_size))
E /= E.max()
print(model(c_1_in).size())
ds = TensorDataset(c_1_in.float(), c_out_l.float())
print("c_1_in.float()",c_1_in.float().shape)
print("c_out_l.float()",c_out_l.float().shape)
dl = DataLoader(ds, batch_size=1, shuffle=True)
trainer = pl.Trainer(accelerator="auto", max_epochs=epoch_1, logger=wandb_logger,deterministic=True)
#trainer = pl.Trainer(accelerator="auto", max_epochs=epoch,deterministic=True)
trainer.fit(model, dl)
trainer.test(model, dl)
c_conv = model(c_1_in)
c_conv_results[n_1_out] = c_conv.detach().numpy()
print("c_conv_results",c_conv.detach().numpy().shape)
E = model.net.E[0]
t_E = torch.linspace(0, float(t_e_1), int(model.kernel_size))
E = E / E.max()

t_1_in_reshaped = t_1_in.squeeze().numpy()
t_conv_l_reshaped = t_conv_l.squeeze().numpy()  # Reshaped for plotting compatibility
c_1_in_reshaped = c_1_in[0, 0, :].squeeze().numpy()
t_conv_reshaped = t_conv[0, 0, :].squeeze().numpy()
c_out_l_reshaped = c_out_l[0, 0, :].squeeze().numpy()
c_conv_reshaped = c_conv[0, 0, :].detach().squeeze().numpy()

# Plotting
fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)
ax1.plot(t_1_in_reshaped, c_1_in_reshaped, label="Input Signal", color="blue")
ax1.plot(t_conv_reshaped, c_out_l_reshaped, label="Expected Output", color="green")
ax1.plot(t_conv_reshaped, c_conv_reshaped, label="Predicted Output", color="red", linestyle="--")
ax1.plot(t_E, E, label="Convolution Kernel", color="purple", linestyle="--")
# ax1.plot(t_conv_l, E_laminar, color="grey")
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.set_xlabel('Time')
ax1.legend()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"1st_Layer_Laminar_Model_1_n_1_out_{disc_n_1_out}_{timestamp}_model3.png", format="png", dpi=300)
plt.show()
wandb.finish()
###############################################################################
###CNN for the Output of the AdlerCNN(middle)
# c_conv_l_3_results = {}
# t_1_l_in = torch.linspace(0, t_lam, n_1_out).unsqueeze(0).unsqueeze(0).float()
# c_1_l_in = torch.zeros((1, 1, n_1_out)).float()
# indices = (t_1_l_in > 5).squeeze()
# c_1_l_in[:, :, indices] = 1
# print("c_1_in shape:", c_1_l_in.shape)  
# print("t_1_in shape:", t_1_l_in.shape) 
# c_out_l_l = torch.tensor(c_out_l_3).unsqueeze(0).unsqueeze(0)  
# t_conv_l_l = torch.tensor(t_conv_l_3).unsqueeze(0).unsqueeze(0) 
# c_out_l_3 = torch.cat([torch.tensor(c_out_l_l)], dim=0).float()
# t_conv_l_3 = torch.cat([torch.tensor(t_conv_l_l)], dim=0).float()
# c_out_3_list = [c_out_l_3]
# t_conv_3_list = [t_conv_l_3]
# c_out_l_3 = torch.cat(c_out_3_list, dim=-1)
# t_conv_l_3 = torch.cat(t_conv_3_list, dim=-1)
# print("c_out shape:", c_out_l_3.shape)  
# print("t_conv shape:", t_conv_l_3.shape) 
 
# model_3 = RTDModule(
#         kernel_size=n_e_2,
#         learning_rate=learning_rate,
#         use_scheduler=True,
#         scheduler_kwargs={"factor": 0.5, "patience": 80},
#     )

# c_conv_l_3= model_3(c_1_l_in)
# print("c_1_l_in",c_1_l_in.shape)
# print("c_conv_l",c_conv_l_3.shape)
# E_3 = model_3.net.E[0]
# t_E_3 = torch.linspace(0, float(t_e_2), int(model_3.kernel_size))
# E_3 /= E_3.max()
# print(model_3(c_1_l_in).size())
# ds_3 = TensorDataset(c_1_l_in.float(), c_out_l_3.float())
# print("c_1_l_in.float()",c_1_l_in.float().shape)
# print("c_out_l.float()",c_out_l_3.float().shape)
# dl_3 = DataLoader(ds_3, batch_size=1, shuffle=True)
# trainer = pl.Trainer(accelerator="auto", max_epochs=epoch_3,deterministic=True)
# trainer.fit(model_3, dl_3)
# trainer.test(model_3, dl_3)
# c_conv_l_3 = model_3(c_1_l_in)
# c_conv_l_3_results[n_2_out] = c_conv_l_3.detach().numpy()
# print("c_conv_l_3_results",c_conv_l_3.detach().numpy().shape)
# E_3 = model_3.net.E[0]
# t_E_3 = torch.linspace(0, float(t_e_2), int(model_3.kernel_size))
# E_3 = E_3 / E_3.max()

# t_1_in_l_reshaped = t_1_l_in.squeeze().numpy()
# t_conv_l_3_reshaped = t_conv_l_3.squeeze().numpy()  # Reshaped for plotting compatibility
# c_1_l_in_reshaped = c_1_l_in[0, 0, :].squeeze().numpy()
# t_conv_l_3_reshaped = t_conv_l_3[0, 0, :].squeeze().numpy()
# c_out_l_3_reshaped = c_out_l_3[0, 0, :].squeeze().numpy()
# c_conv_l_3_reshaped = c_conv_l_3[0, 0, :].detach().squeeze().numpy()

# # Plotting
# fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(10, 8))
# fig.subplots_adjust(hspace=0)
# ax1.plot(t_1_in_l_reshaped, c_1_l_in_reshaped, label="Input Signal", color="blue")
# ax1.plot(t_conv_l_3_reshaped, c_out_l_3_reshaped, label="Expected Output", color="green")
# ax1.plot(t_conv_l_3_reshaped, c_conv_l_3_reshaped, label="Predicted Output", color="red", linestyle="--")
# ax1.plot(t_E_3, E_3, label="Convolution Kernel", color="purple", linestyle="--")
# # ax1.plot(t_conv_l, E_laminar, color="grey")
# ax1.set_xlim((0, 30))
# ax1.set_ylim((0, 1.1))
# ax1.set_ylabel('Concentration')
# ax1.set_xlabel('Time')
# ax1.legend()
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plt.savefig(f"Model_3_1st_Layer_Laminar_For_2nd_Layer_n_1_out_{disc_n_1_out}_{timestamp}_model3.png", format="png", dpi=300)
# plt.show()

###############################################################################

#Adler Model, with the 1st layer's output as input for the second model 
# def compute_inverse_laplace(coefficients, t_values):
#     s, t = sp.symbols('s t')
#     alpha = coefficients['alpha_val']
#     results = []  
#     for tau_a_val in coefficients['tau_a_val']:  
#         for tau_p_val in coefficients['tau_p_val']:
#             for beta_val in coefficients['beta_val']:
#                 tau_m_val = (beta_val * (1 - alpha)) / alpha
#                 F_s = (sp.exp(-tau_p_val * s)) / (1 + beta_val + tau_a_val * s - (beta_val / (1 + tau_m_val * s)))
#                 f_t = sp.inverse_laplace_transform(F_s, s, t)
#                 f_t_numeric = sp.lambdify(t, f_t, modules="numpy")
#                 E_t = f_t_numeric(t_values)
#                 results.append({
#                     'tau_a_val': tau_a_val,
#                     'tau_p_val': tau_p_val,
#                     'tau_m_val': tau_m_val,
#                     'beta_val': beta_val,
#                     'E_t': E_t
#                 })
#     return results

# t_values_Adler = np.linspace(0, t_adl, n_2_out, endpoint=True)
# results = compute_inverse_laplace(coefficients, t_values_Adler)
# # adler_dir = os.path.join(base_dir, f'Adler_tau_a')
# # os.makedirs(adler_dir, exist_ok=True)

# for result in results:
#     tau_a_val = result['tau_a_val']
#     tau_p_val = result['tau_p_val']
#     tau_m_val = result['tau_m_val']
#     beta_val = result['beta_val']
    
#     E_Adler = result['E_t']
#     if E_Adler.shape[0] != t_values_Adler.shape[0]:
#         if E_Adler.shape[0] > t_values_Adler.shape[0]:
#             E_Adler = E_Adler[:len(t_values_Adler)]
#         else:
#             E_Adler = np.pad(E_Adler, (0, len(t_values_Adler) - len(E_Adler)), 'constant')
#         E_Adler=E_Adler/E_Adler.max()
#     E_Adler_normalized=E_Adler/np.sum(E_Adler)
#     # print(f"Shape of E_Adler_normalized: {E_Adler_normalized.shape}")


# c_out_full_Adler = np.convolve(c_conv_l_3_reshaped, E_Adler_normalized, mode="full")
# t_conv_full_Adler = np.linspace(t_values_Adler[0] + t_values_Adler[0], t_values_Adler[-1] +t_values_Adler[-1], len(c_out_full_Adler))
# valid_indices = t_conv_full_Adler <= t_adl # Adjust time limit as needed
# t_conv_Adler = t_conv_full_Adler[valid_indices]
# c_out_Adler = c_out_full_Adler[valid_indices]
# #Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(t_values_Adler, c_conv_l_3_reshaped, label="Input_Laminar_NN", color="blue")
# plt.plot(t_conv_Adler, c_out_Adler, label="Output from the model", color="orange")
# plt.xlabel("Time", fontsize=14)
# plt.ylabel("conc", fontsize=14)
# plt.legend(fontsize=12)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plt.savefig(f"Adler_model_and_CNN_Laminar_model_n_1_out_{disc_n_1_out}_{timestamp}_model3.png", format="png", dpi=300)
# plt.show()

# print(f"Shape of t_conv_Adler: {t_conv_Adler.shape}")
# print(f"Shape of c_out_Adler: {c_out_Adler.shape}")
# print(f"Shape of t_values_Adler: {t_values_Adler.shape}")
# print(f"Shape of E_Adler_normalized: {E_Adler_normalized.shape}")
# print(f"Shape of c_conv for discretization: {c_conv.shape}")
# print(f"Shape of c_conv_reshaped for discretization: {c_conv_reshaped.shape}")
##############################################################################
##Second CNN, the input is coming from the NN and the predicted output(error) is coming from the theory
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True,name=f'Model_2_n_2_out_{n_out_1}_learning_rate_{learning_rate}_2nd_epoch_{epoch_2}',

    reinit=True
)
c_conv_2_results = {}
t_2_in=torch.tensor(t_conv_l).float()
c_2_in=torch.tensor(c_out_l).float()
print("c_out_l",c_out_l.shape)
c_out_l_a=torch.tensor(c_out_Adler).float().unsqueeze(0).unsqueeze(0)
t_out_l_a=torch.tensor(t_conv_Adler).float().unsqueeze(0).unsqueeze(0)
print(t_2_in.shape)
print(c_2_in.shape)
print(c_out_l_a.shape)
print(t_out_l_a.shape)
# Initialize the model
model_2 = RTDModule(
    kernel_size=n_e_2,
    learning_rate=learning_rate,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)
c_conv_2 = model_2(c_2_in)
E = model_2.net.E[0]
t_E_2 = torch.linspace(0, t_e_2, model_2.kernel_size)
E = E / E.max()
ds_2 =TensorDataset(c_2_in.float(), c_out_l_a.float())
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
c_conv_2 = model_2(c_2_in)
E_2 = model_2.net.E[0]
c_conv_2_results[n_2_out] = c_conv_2.detach().numpy()
#E = model_2.net.E[0]
t_E_2 = torch.linspace(0, t_e_2, model_2.kernel_size)
E_2 = E_2 / E_2.max()
####Plotting
fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(10, 8))
ax1.plot(t_2_in.squeeze().numpy(), c_2_in.squeeze().numpy(), label="Input Signal", color="blue", linestyle="--")
ax1.plot(t_out_l_a.squeeze().numpy(), c_out_l_a.squeeze().numpy(), label="Expected Output", color="green")
ax1.plot(t_out_l_a.squeeze().numpy(), c_conv_2.detach().squeeze().numpy(), label="Predicted Output", color="red", linestyle="--")
ax1.plot(t_E_2, E_2, label="E_predict", color="purple", linestyle="--")
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.set_xlabel('Time')
ax1.legend()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"Model_2_2nd_Layer_Adler_theory_based_n_1_out_{disc_n_1_out}_{timestamp}.png", format="png", dpi=300)
plt.show()
################################################################################
##Second CNN, the input and the predicted output both is coming from the NN 
# wandb_logger = pl_loggers.WandbLogger(
#     project="nRTD",
#     log_model=True,name=f'Model_4_n_1_out_{disc_n_1_out}_learning_rate_{learning_rate}_2nd_epoch_{epoch_2}',
#     reinit=True
# )
# c_conv_2_results = {}
# t_2_in=torch.tensor(t_conv_l).float()
# c_2_in=torch.tensor(c_out_l).float()
# print("c_out_l",c_out_l.shape)
# c_out_l_a=torch.tensor(c_out_Adler).float().unsqueeze(0).unsqueeze(0)
# t_out_l_a=torch.tensor(t_conv_Adler).float().unsqueeze(0).unsqueeze(0)
# print(t_2_in.shape)
# print(c_2_in.shape)
# print(c_out_l_a.shape)
# print(t_out_l_a.shape)
# # Initialize the model
# model_2 = RTDModule(
#     kernel_size=n_e_2,
#     learning_rate=learning_rate,
#     use_scheduler=True,
#     scheduler_kwargs={"factor": 0.5, "patience": 80},
# )
# c_conv_2 = model_2(c_2_in)
# E = model_2.net.E[0]
# t_E_2 = torch.linspace(0, t_e_2, model_2.kernel_size)
# E = E / E.max()
# ds_2 =TensorDataset(c_2_in.float(), c_out_l_a.float())
# dl_2 = DataLoader(ds_2, batch_size=1, shuffle=True)

# trainer = pl.Trainer(
#     accelerator="auto",
#     max_epochs=epoch_2,
#     logger=wandb_logger,
#     deterministic=True,
# )
# # trainer = pl.Trainer(
# #     accelerator="auto",
# #     max_epochs=epoch,
# #     deterministic=True,
# # )
# trainer.fit(model_2, dl_2)
# trainer.test(model_2, dl_2)
# c_conv_2 = model_2(c_2_in)
# E_2 = model_2.net.E[0]
# c_conv_2_results[n_2_out] = c_conv_2.detach().numpy()
# #E = model_2.net.E[0]
# t_E_2 = torch.linspace(0, t_e_2, model_2.kernel_size)
# E_2 = E_2 / E_2.max()
# ####Plotting
# fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(10, 8))
# ax1.plot(t_2_in.squeeze().numpy(), c_2_in.squeeze().numpy(), label="Input Signal", color="blue", linestyle="--")
# ax1.plot(t_out_l_a.squeeze().numpy(), c_out_l_a.squeeze().numpy(), label="Expected Output", color="green")
# ax1.plot(t_out_l_a.squeeze().numpy(), c_conv_2.detach().squeeze().numpy(), label="Predicted Output", color="red", linestyle="--")
# ax1.plot(t_E_2, E_2, label="E_predict", color="purple", linestyle="--")
# ax1.set_xlim((0, 30))
# ax1.set_ylim((0, 1.1))
# ax1.set_ylabel('Concentration')
# ax1.set_xlabel('Time')
# ax1.legend()
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plt.savefig(f"Model_2_2nd_Layer_Adler_CNN_based_{timestamp}_model3_n_1_out_{disc_n_1_out}.png", format="png", dpi=300)
# plt.show()
################################################################################
###Second NN, the input is coming from the theory
# wandb_logger = pl_loggers.WandbLogger(
#     project="nRTD",
#     log_model=True,name="input as theo lam 2nd",
#     reinit=True
# )
# c_conv_2_results = {}
# t_2_in=t_conv_l.float()
# c_2_in=c_out_l.float()
# print("c_out_l",c_out_l.shape)
# c_out_l_a=torch.tensor(c_out_Adler).float().unsqueeze(0).unsqueeze(0)
# t_out_l_a=torch.tensor(t_conv_Adler).float().unsqueeze(0).unsqueeze(0)
# print(t_2_in.shape)
# print(c_2_in.shape)
# print(c_out_l_a.shape)
# print(t_out_l_a.shape)
# # Initialize the model
# model_2 = RTDModule(
#     kernel_size=n_e_2,
#     learning_rate=learning_rate,
#     use_scheduler=True,
#     scheduler_kwargs={"factor": 0.5, "patience": 80},
# )
# c_conv_2 = model_2(c_2_in)
# E = model_2.net.E[0]
# t_E_2 = torch.linspace(0, t_e_2, model_2.kernel_size)
# E = E / E.max()
# ds_2 =TensorDataset(c_2_in.float(), c_out_l_a.float())
# dl_2 = DataLoader(ds_2, batch_size=1, shuffle=True)

# trainer = pl.Trainer(
#     accelerator="auto",
#     max_epochs=epoch_2,
#     logger=wandb_logger,
#     deterministic=True,
# )
# # trainer = pl.Trainer(
# #     accelerator="auto",
# #     max_epochs=epoch,
# #     deterministic=True,
# # )
# trainer.fit(model_2, dl_2)
# trainer.test(model_2, dl_2)
# c_conv_2 = model_2(c_2_in)
# E_2 = model_2.net.E[0]
# c_conv_2_results[n_2_out] = c_conv_2.detach().numpy()
# #E = model_2.net.E[0]
# t_E_2 = torch.linspace(0, t_e_2, model_2.kernel_size)
# E_2 = E_2 / E_2.max()
# ####Plotting
# fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(10, 8))
# ax1.plot(t_2_in.squeeze().numpy(), c_2_in.squeeze().numpy(), label="Input Signal", color="blue")
# ax1.plot(t_out_l_a.squeeze().numpy(), c_out_l_a.squeeze().numpy(), label="Expected Output", color="green")
# ax1.plot(t_out_l_a.squeeze().numpy(), c_conv_2.detach().squeeze().numpy(), label="Predicted Output", color="red", linestyle="--")
# ax1.plot(t_E_2, E_2, label="E_predict", color="purple", linestyle="--")
# ax1.set_xlim((0, 30))
# ax1.set_ylim((0, 1.1))
# ax1.set_ylabel('Concentration')
# ax1.set_xlabel('Time')
# ax1.legend()
#### General Plotting
E_Adler_normalized = E_Adler / np.max(E_Adler)
E_learned_1 = model.net.E[0] if isinstance(model.net.E[0], np.ndarray) else model.net.E[0].numpy()
E_learned_2 = model_2.net.E[0] if isinstance(model_2.net.E[0], np.ndarray) else model_2.net.E[0].numpy()
E_learned_1 /= np.max(E_learned_1)
E_learned_2 /= np.max(E_learned_2)
t_adler = t_values_Adler if isinstance(t_values_Adler, np.ndarray) else t_values_Adler.numpy()
t_learned_1 = np.linspace(0, t_e, len(E_learned_1))
t_learned_2 = np.linspace(0, t_e_2, len(E_learned_2))

plt.figure(figsize=(10, 6))
plt.plot(t_learned_1, E_learned_1, label="$E_1$", color="black")
plt.plot(t_learned_2, E_learned_2, label="$E_2$", color="red")
# plt.plot(t_l, E_laminar, label="E Laminar Layer 1", color="blue")
#plt.plot(t_l_a, E_laminar_a_normalized, label="E Laminar Layer 2", color="green")
# plt.plot(t_adler, E_Adler_normalized, label="E Adler", color="orange")
print("t_learned_1",t_learned_1.shape)
print("t_learned_2",t_learned_2.shape)
print("E_learned_1", E_learned_1.shape)
print("E_learned_2", E_learned_2.shape)
# Set plot labels and legend
plt.xlabel("Time")
plt.ylabel("E")
plt.xlim((0, 20))
plt.ylim((0, 1.1))
plt.legend()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"E_n_1_out_{disc_n_1_out}_{timestamp}_model3.png", format="png", dpi=300)
plt.show()
#Plotting the test/loss
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


fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(t_1_in_reshaped, c_1_in_reshaped, label=" $c_{in}$", color="blue")
ax1.plot(t_conv[0, 0, :].squeeze().numpy(), c_conv[0, 0, :].detach().squeeze().numpy(), label="$c_{p,1}$", color="black", linestyle="--")
ax1.plot(t_conv[0, 0, :].squeeze().numpy(),c_out_l[0, 0, :].squeeze().numpy() , label="$c_{o,1}$", color="yellow")
ax1.plot(t_out_l_a.squeeze().numpy(), c_conv_2.detach().squeeze().numpy(), label="$c_{p,2}$", color="red", linestyle="--")
ax1.plot(t_out_l_a.squeeze().numpy(), c_out_l_a.squeeze().numpy(), label="$c_{o,2}$", color="orange")
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.set_xlabel('Time')
ax1.legend()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"Conc_profiles__n_1_out_{disc_n_1_out}_{timestamp}_model3.png", format="png", dpi=300)
plt.show()