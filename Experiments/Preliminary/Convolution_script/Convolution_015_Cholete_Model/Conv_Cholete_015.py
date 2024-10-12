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
from nrtd import RTDModule
import numpy.typing as npt

Bo_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Cholete_Model/beta0.9'
t_conv_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

n_disc = 119
t_input = torch.linspace(0, 40, n_disc)
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
    kernel_size=80,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 60, model.kernel_size)
E = E / E.max()
j = 0  

plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label= "beta0.9",
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
    max_epochs=10000, deterministic=True
)

trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 60, model.kernel_size)
E = E / E.max()

def Cholete(t: npt.NDArray[np.float64], alpha: float, beta: float, tau: float, g: float) -> npt.NDArray[np.float64]: 
    shifted_t = t
    shifted_t[shifted_t < 0] = 0  
    H = np.where(shifted_t < g, 0, 1)
    k = ((1 - alpha) / (beta * tau))
    exp_term = (1 - alpha) * np.exp(k * (g - shifted_t))
    F = alpha * H - exp_term + (1 - alpha)
    F[F < 0] = 0  
    E_expected = np.gradient(F, t)  
    E_expected = E_expected / E_expected.max()  
    return F, E_expected

alpha = 0.2
beta = 0.9
tau = 5
g = 2.5
t_values = torch.linspace(0, 60, 80).numpy()

F, E_expected = Cholete(t_values, alpha, beta, tau, g)



plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

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
ax1.set_xlim((0, 150))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.legend()
ax1.tick_params(labelbottom=False)
ax2.plot(t_E+5, E, label="E", color="orange")
ax2.set_xlabel('t')
ax2.set_ylabel('E')
ax2.plot(t_values, E_expected, label="E (Expected )", color="purple", linestyle="--")
ax2.set_xlim((0, 150))
ax2.set_ylim((0, 1.1))
ax2.legend()
ax2.set_xticks(np.arange(0, 10, 1))  
plt.xlabel("Time")
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_015_Cholete_Model"
unified_dir = os.path.join(save_dir, f'Bo_{0.9}')
os.makedirs(unified_dir, exist_ok=True)
plt.savefig(os.path.join(unified_dir, 'Figure_Cholete_changed.png'), dpi=300)
plt.show()

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
plt.plot(predicted_time+4, predicted_E, label='E_predicted_Bo9', color='orange')
plt.plot(expected_time, expected_E, label='E_expected_Bo9', color='purple', linestyle='--')
plt.xlabel('t')
plt.ylabel('E')
plt.xlim((predicted_time.min(), predicted_time.max()))
plt.ylim((0,1.1 ))  
plt.xlim(0,25)
plt.legend()
plt.xticks(np.arange(0, 10, 1))  
plt.savefig(os.path.join(unified_dir, 'E_saved_12102024_Bo_09_changed.png'), dpi=300)
plt.show()



