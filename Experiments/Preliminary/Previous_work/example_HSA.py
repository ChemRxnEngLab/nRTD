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
from nrtd import RTDModule
from lightning.pytorch import loggers as pl_loggers
import sympy as sp
import os
import datetime

n_0_in=100
n_0_E=99
n_0_out=200
n_1_in=n_0_out
n_1_E=132
n_1_out=334
t_lam=60
t_adl=50

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
t_l = np.linspace(0, t_lam, n_0_out, endpoint=True)  
c_0_l = np.zeros_like(t_l)
c_0_l[t_l > 5] = 1  
E_laminar = laminarflow(t_l, tau_l)
E_laminar = E_laminar / E_laminar.max()
c_out_l_full = np.convolve(c_0_l, E_laminar / np.sum(E_laminar), mode="full")
t_conv_l_full = np.linspace(t_l[0] + t_l[0], t_l[-1] + t_l[-1], len(c_out_l_full))
valid_indices = t_conv_l_full <= 60
t_conv_l = t_conv_l_full[valid_indices]
c_out_l = c_out_l_full[valid_indices]

### Adler Model###
t_values_Adler = np.linspace(0, t_adl, n_1_out, endpoint=True)
results = compute_inverse_laplace(coefficients, t_values_Adler)

def compute_inverse_laplace(coefficients, t_values):
    s, t = sp.symbols('s t')
    alpha = coefficients['alpha_val']
    results = []  
    for tau_a_val in coefficients['tau_a_val']:  
        for tau_p_val in coefficients['tau_p_val']:
            for beta_val in coefficients['beta_val']:
                tau_m_val = (beta_val * (1 - alpha)) / alpha
                F_s = (sp.exp(-tau_p_val * s)) / (1 + beta_val + tau_a_val * s - (beta_val / (1 + tau_m_val * s)))
                print("E(s)")
                sp.pprint(F_s)
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
                print("Inverse Laplace transform E(t):")
                sp.pprint(f_t)
                integral_E_t = np.trapz(E_t, t_values)
                print(f"Integral of E(t) over the time range: {integral_E_t}")
    return results


adler_dir = os.path.join(base_dir, f'Adler_tau_a_{coefficients["tau_a_val"][0]}_tau_p_{coefficients["tau_p_val"][0]}_Disc_{disc}')
os.makedirs(adler_dir, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

for result in results:
    tau_a_val = result['tau_a_val']
    tau_p_val = result['tau_p_val']
    tau_m_val = result['tau_m_val']
    beta_val = result['beta_val']
    E_Adler = result['E_t']
    E_Adler_normalized = E_Adler / np.sum(E_Adler)
def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E_laminar_a = np.zeros_like(t)
    E_laminar_a[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E_laminar_a

t_l_a = np.linspace(0, 60, disc, endpoint=True)  
c_0_l_a = np.zeros_like(t_l)
c_0_l_a[t_l > 5] = 1  
E_laminar_a = laminarflow(t_l_a, tau_l)
E_laminar_a = E_laminar_a / E_laminar_a.max()
c_out_l_a_full = np.convolve(c_0_l_a, E_laminar_a / np.sum(E_laminar_a), mode="full")
t_conv_l_a_full = np.linspace(t_l_a[0] + t_l_a[0], t_l_a[-1] + t_l_a[-1], len(c_out_l_full))
valid_indices = t_conv_l_a_full <= 60
t_conv_l_a = t_conv_l_a_full[valid_indices]
c_out_l_a = c_out_l_a_full[valid_indices]
c_0_Adler = c_out_l_a 

c_out_full_Adler = np.convolve(c_0_Adler_normalized, E_Adler_normalized, mode="full")
t_conv_full_Adler = np.linspace(t_values_Adler[0] * 2, t_values_Adler[-1] * 2, len(c_out_full_Adler))
valid_indices = t_conv_full_Adler <= t_adl #TIME BE CAREFUL
t_conv_Adler = t_conv_full_Adler[valid_indices]
c_out_Adler = c_out_full_Adler[valid_indices]



np.save(os.path.join(adler_dir, f'time_Adler_disc_{disc}.npy'), t_conv_Adler)
np.save(os.path.join(adler_dir, f'concentration_Adler_disc_{disc}.npy'), c_out_Adler)
ax1.plot(t_values_Adler, E_Adler_normalized, label=f'tau_a={tau_a_val}, tau_p={tau_p_val}, beta={beta_val}')
ax2.plot(t_conv_Adler, c_out_Adler, label=f'tau_a={tau_a_val}, tau_p={tau_p_val}, beta={beta_val}')

ax1.set_xlabel('Time')
ax1.set_ylabel('E(t)')
ax1.legend()
ax2.plot(t_values_Adler, c_0_Adler_normalized, label='c_0', linestyle='--', color='black')
ax2.set_xlim(0, 50)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('Time')
ax2.set_ylabel('Concentration C')
ax2.legend()
plt.savefig(os.path.join(adler_dir, f'Adler_Model_Disc_{disc}.png'), dpi=300)
plt.show()
print(f"Shape of c_out_Adler for discretization {disc}: {c_out_Adler.shape}")


c_1 = np.convolve(lam_flow_model, c_0_in)
## len(c_1)= n_0_out

c_2 = np.convolve(Adler_havarka, c_1)
## len(c_2)= n_1_out

### Neural Network
#input to the first NNlayer
c_0_in = np.zeros((n_0_in,))
c_0_in[t_0_in>5] = 1

model_1= RTDModule(
    kernel_size=n_0_E,
    ...
)

ds_1 = TensorDataset(torch.tensor(c_0_in), torch.tensor(c_1))

Triner.train(model_1, ds_1)

model_1.E
plt.plot()

c_1_tilde = model_1(torch.tensor(c_0_in))

#input to the second NNlayer

model_2 = RTDModule(
    kernel_size=n_1_E,
    ...
)

ds_2 = TensorDataset(c_1_tilde, torch.tensor(c_2))

Trainer.train(model_2, ds_2)

model2.E
plt.plot()