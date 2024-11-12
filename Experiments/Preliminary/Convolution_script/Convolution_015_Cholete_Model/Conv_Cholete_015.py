import sys
import os
import torch
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import matplotlib.pyplot as plt
import numpy as np
import wandb
from nRTD import RTDModule
import numpy.typing as npt

Bo_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Cholete_Model/beta0.1'
t_conv_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
epoch=10000
n_disc = 100
t_input = torch.linspace(0, 70, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 5] = 1
c_in[1::2, :, t_input < 5] = 1

c_out_list = []
t_conv_list = []
c_out_list.append(c_out_tau)
t_conv_list.append(t_conv_tau)

c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

model = RTDModule(
    kernel_size=99,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 70, model.kernel_size)
E = E / E.max()
j = 0  

plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label= "beta0.1",
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
plt.ylim((0,1.1))
plt.show()

print(model(c_in).size())

ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=epoch, deterministic=True
)

trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 70, model.kernel_size)
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
    
t_exp = np.linspace(0, 140, 99)
c_0_exp = np.zeros_like(t_exp)
c_0_exp[t_exp > 5] = 1
beta_values = np.array([0.1])
for beta in beta_values:
    F,E_e = Cholete(t_exp, 0.2, beta, 5,5)
    E_c_e = Cholete_E(t_exp, 0.2, beta, 5)
    #E_t_normalized = E / np.sum(E)
    E_c_e_normalized = E_c_e / np.sum(E_c_e)
    #print(f"beta: {beta}, Integral of E: {np.sum(E)}")
    c_out_full = np.convolve(c_0_exp, E_c_e_normalized, mode="full")
    t_conv_full = np.linspace(t_exp[0] + t_exp[0], t_exp[-1] + t_exp[-1], len(c_out_full))
    valid_indices = t_conv_full <= 140
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
ax1.set_xlim((0, 8))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.set_xticks(np.arange(0, 7, 1)) 
ax1.legend()
ax1.tick_params(labelbottom=False)
ax2.plot(t_E, E, label="E", color="orange")
ax2.plot(t_conv_e, E_c_e_normalized/E_c_e_normalized.max(), label="E", color="red")
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
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_015_Cholete_Model"
unified_dir = os.path.join(save_dir, f'Bo_{0.9}')
os.makedirs(unified_dir, exist_ok=True)
plt.savefig(os.path.join(unified_dir, 'Figure_Cholete_09.png'), dpi=300)
plt.show()
print("E",E.shape)
print("E_c_e_normalized",E_c_e_normalized.shape)
#####




plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)  
#ax1.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")


# for i in range(c_out.size(1)):
#     ax1.plot(
#         t_conv[0, i, :].numpy(),
#         c_out[0, i, :].numpy(),
#         label="Bo=0.9",
#         color="green",
#     )


# ax1.plot(
#     t_conv[0, 0, :].numpy(),
#     c_conv[0, 0, :].detach().numpy(),
#     label="Predicted",
#     color="red",linestyle="-."
# )
ax1.set_xlim((0, 8))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.set_xticks(np.arange(0, 7, 1)) 
ax1.legend()
#ax1.tick_params(labelbottom=False)
ax2.plot(t_E, E, label="E", color="orange")
ax2.plot(t_conv, E_c_normalized, label="E", color="red")
#ax2.plot(t_values, E_expected, label="E (Expected )", color="purple", linestyle="--")
ax2.set_xlabel('t')
ax2.set_ylabel('E')
#ax2.plot(t_values, E_expected, label="E (Expected )", color="purple", linestyle="--")
#ax2.plot(t_conv[0, i, :].numpy(), E, label="E (Expected )", color="purple", linestyle="--")
ax2.set_xlim((0, 50))
ax2.set_ylim((0, 1.1))
ax2.legend()
ax2.set_xticks(np.arange(0, 10, 1))  
plt.xlabel("Time")

#plt.savefig(os.path.join(unified_dir, 'Figure_Cholete_09.png'), dpi=300)
plt.show()






####
predicted_E = E
predicted_time = t_E              
expected_E = E_expected                 
expected_time = t_values
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_015_Cholete_Model"
np.save(os.path.join(unified_dir, 'E_predicted_Bo_9.npy'), predicted_E)
np.save(os.path.join(unified_dir, 't_E_predicted_Bo_9.npy'), predicted_time)
np.save(os.path.join(unified_dir, 'E_expected_Bo_9.npy'), expected_E)
np.save(os.path.join(unified_dir, 't_E_expected_Bo_9.npy'), expected_time)
print("saved under:", unified_dir)
print("E_predicted",predicted_E)
print("E_expected",expected_E)
predicted_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_015_Cholete_Model/Bo_0.9/E_predicted_Bo_9.npy')
predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_015_Cholete_Model/Bo_0.9/t_E_predicted_Bo_9.npy')
plt.plot(expected_time, predicted_E, label='E_predicted_Bo3', color='orange')
plt.plot(expected_time, expected_E, label='E_expected_Bo3', color='purple', linestyle='--')
plt.xlabel('t')
plt.ylabel('E')
plt.xlim((predicted_time.min(), predicted_time.max()))
plt.ylim((0,1.1 ))  
plt.xlim(0,25)
plt.legend()
plt.xticks(np.arange(0, 10, 1))  
plt.savefig(os.path.join(unified_dir, 'E_saved_12102024_Bo_3.png'), dpi=300)
plt.show()
print("expected_time",expected_time.shape)
print("predicted_time",predicted_time.shape)