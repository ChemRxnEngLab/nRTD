import matplotlib.pyplot as plt
import numpy.typing as npt
import sys
import os
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import numpy as np
import wandb
from nRTD import RTDModule
from lightning.pytorch import loggers as pl_loggers
import sympy as sp
import os
import datetime
base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/2_nd_Layer_Convolution'

n_1_in=100
n_1_E=99
n_1_out=200
n_2_in=n_1_out
n_2_E=133
n_2_out=334
t_lam=60
t_adl=50
t_1=30
t_e=30
t_e_2=25

#Parameters
tau_l = 5.0
coefficients = {
    'tau_a_val': np.array([1]),   
    'tau_p_val': np.array([2]), 
    'beta_val': np.array([0.1]), 
    'alpha_val': 0.2           
}

### Model Laminar, 200disc###
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
                #print("Inverse Laplace transform E(t):")
                #print(E_t)
    return results

t_values_Adler = np.linspace(0, t_adl, n_2_out, endpoint=True)
results = compute_inverse_laplace(coefficients, t_values_Adler)
adler_dir = os.path.join(base_dir, f'Adler_tau_a')
os.makedirs(adler_dir, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

for result in results:
    tau_a_val = result['tau_a_val']
    tau_p_val = result['tau_p_val']
    tau_m_val = result['tau_m_val']
    beta_val = result['beta_val']
    
    E_Adler = result['E_t']
    if E_Adler.shape[0] != t_values_Adler.shape[0]:
        print(f"Warning: E_Adler has shape {E_Adler.shape}, adjusting to match t_values_Adler shape.")
        if E_Adler.shape[0] > t_values_Adler.shape[0]:
            E_Adler = E_Adler[:len(t_values_Adler)]
        else:
            E_Adler = np.pad(E_Adler, (0, len(t_values_Adler) - len(E_Adler)), 'constant')

    E_Adler_normalized=E_Adler/np.sum(E_Adler)
    print(f"Shape of t_values_Adler: {t_values_Adler.shape}")
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
print(f"Shape of c_out_l_a: {c_out_l_a.shape}")

###
c_out_full_Adler = np.convolve(c_out_l_a, E_Adler_normalized, mode="full")
t_conv_full_Adler = np.linspace(t_values_Adler[0] + t_values_Adler[0], t_values_Adler[-1] +t_values_Adler[-1], len(c_out_full_Adler))
valid_indices = t_conv_full_Adler <= t_adl # Adjust time limit as needed
t_conv_Adler = t_conv_full_Adler[valid_indices]
c_out_Adler = c_out_full_Adler[valid_indices]

# Check shapes before plotting
print(f"Shape of t_conv_Adler: {t_conv_Adler.shape}")
print(f"Shape of c_out_Adler: {c_out_Adler.shape}")
print(f"Shape of t_values_Adler: {t_values_Adler.shape}")
print(f"Shape of E_Adler_normalized: {E_Adler_normalized.shape}")
print(f"Shape of c_out_l_a for discretization: {c_out_l_a.shape}")
print(f"Shape of t_conv_l_a for discretization: {t_conv_l_a.shape}")

#NN
c_conv_results = {}
# t_1_in = torch.linspace(0, t_1, n_1_in).unsqueeze(0).unsqueeze(0)  #
# c_1_in = torch.zeros((1, 1, n_1_in))
# c_1_in[:, :, t_1_in > 5] = 1
# print("c_1_in",c_1_in.shape)
# print("t_1_in",t_1_in.shape)


# t_1_in = torch.linspace(0, t_1, n_1_in).unsqueeze(0).unsqueeze(0)  # Shape will be [1, 1, 100]
# c_1_in = torch.zeros((1, 1, n_1_in))
# indices = (t_1_in > 5).squeeze()
# c_1_in[:, :, indices] = 1


t_1_in = torch.linspace(0, t_1, n_1_in).unsqueeze(0).unsqueeze(0).float()
c_1_in = torch.zeros((1, 1, n_1_in)).float()
indices = (t_1_in > 5).squeeze()
c_1_in[:, :, indices] = 1



print("c_1_in shape:", c_1_in.shape)  
print("t_1_in shape:", t_1_in.shape) 

# c_out_list = [torch.tensor(c_out_l)]  

# # Concatenate tensors in c_out_list along the specified dimension
# t_conv_list = [torch.tensor(t_conv_l)]
# c_out = torch.cat(c_out_list, dim=0)
# t_conv = torch.cat(t_conv_list, dim=0)

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
        kernel_size=n_1_E,
        learning_rate=1e-3,
        use_scheduler=True,
        scheduler_kwargs={"factor": 0.5, "patience": 80},
    )

c_conv = model(c_1_in)
E = model.net.E[0]
t_E = torch.linspace(0, t_e, model.kernel_size)
E /= E.max()
print(model(c_1_in).size())


ds = TensorDataset(c_1_in.float(), c_out_l.float())
print("c_1_in.float()",c_1_in.float().shape)
print("c_out_l.float()",c_out_l.float().shape)

dl = DataLoader(ds, batch_size=20, shuffle=True)
trainer = pl.Trainer(accelerator="auto", max_epochs=1, deterministic=True)
trainer.fit(model, dl)
trainer.test(model, dl)
c_conv = model(c_1_in)
c_conv_results[n_1_out] = c_conv.detach().numpy()
print("c_conv_results",c_conv.detach().numpy().shape)
E = model.net.E[0]
t_E = torch.linspace(0, t_e, model.kernel_size)
E = E / E.max()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)


t_1_in_reshaped = t_1_in.squeeze().numpy()
c_1_in_reshaped = c_1_in[0, 0, :].squeeze().numpy()
t_conv_reshaped = t_conv[0, 0, :].squeeze().numpy()
c_out_l_reshaped = c_out_l[0, 0, :].squeeze().numpy()
c_conv_reshaped = c_conv[0, 0, :].detach().squeeze().numpy()
print("c_conv",c_conv.shape)


ax1.plot(t_1_in_reshaped, c_1_in_reshaped, label="Input Signal", color="blue")
ax1.plot(t_conv_reshaped, c_out_l_reshaped, label="Expected Output", color="green")
ax1.plot(t_conv_reshaped, c_conv_reshaped, label="Predicted Output", color="red", linestyle="--")
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.legend()


c_conv_2_results = {}
model_2 = RTDModule(
        kernel_size=n_2_E,
        learning_rate=1e-3,
        use_scheduler=True,
        scheduler_kwargs={"factor": 0.5, "patience": 80},
    )
c_conv_2 = model(c_conv)
E = model.net.E[0]
t_E = torch.linspace(0, t_e_2, model.kernel_size)
E /= E.max()

c_out_Adler_tensor = torch.tensor(c_out_Adler).float().unsqueeze(0).unsqueeze(0)
ds = TensorDataset(c_conv.float(), c_out_Adler_tensor)
print("c_conv.float()",c_conv.shape)
print("c_out_Adler_tensor",c_out_Adler_tensor.shape)
dl = DataLoader(ds, batch_size=1, shuffle=True)

print("c_2_in shape:", c_conv.shape)  
print("t_2_in shape:", t_conv.shape) 
print("c_2_out shape:", c_out_Adler_tensor.shape) 

trainer = pl.Trainer(accelerator="auto", max_epochs=10000, deterministic=True)
trainer.fit(model, dl)
trainer.test(model, dl)
c_conv_2 = model(c_conv)
c_conv_results[n_2_out] = c_conv_2.detach().numpy()
E = model.net.E[0]
t_E = torch.linspace(0, t_e_2, model.kernel_size)
E = E / E.max()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)


# t_1_in_reshaped = t_1_in.squeeze().numpy()
# c_1_in_reshaped = c_1_in[0, 0, :].squeeze().numpy()
# t_conv_reshaped = t_conv[0, 0, :].squeeze().numpy()
# c_out_l_reshaped = c_out_l[0, 0, :].squeeze().numpy()
# c_conv_reshaped = c_conv[0, 0, :].detach().squeeze().numpy()


ax1.plot(t_conv, c_conv, label="Input Signal", color="blue")
ax1.plot(t_conv_Adler, c_out_Adler, label="Expected Output", color="green")
ax1.plot(t_conv_Adler, c_conv_2, label="Predicted Output", color="red", linestyle="--")
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.legend()

# model_1= RTDModule(
#     kernel_size=n_0_E,
#     ...
# )

# ds_1 = TensorDataset(torch.tensor(c_0_in), torch.tensor(c_1))

# Triner.train(model_1, ds_1)

# model_1.E
# plt.plot()

# c_1_tilde = model_1(torch.tensor(c_0_in))
