import sys
import os
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from nrtd import RTDModule
from lightning.pytorch import loggers as pl_loggers

if wandb.run is not None:
    wandb.finish()
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)

tau_5_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model/tau_200_5.0'
t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

n_disc = 100
t_input = torch.linspace(0, 30, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 5] = 1
c_in[1::2, :, t_input < 5] = 1
c_out_list = []
t_conv_list = []
c_out_list.append(c_out_tau)
t_conv_list.append(t_conv_tau)
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

plt.show()
print(f"c_in size: {c_in.size()}")
print(f"c_out size: {c_out.size()}")
print(f"t_conv size: {t_conv.size()}")

model = RTDModule(
    kernel_size=99,
    learning_rate=10e-4,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 30, model.kernel_size)
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
    log_model=True
)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000,
    logger=wandb_logger,
    deterministic=True,
)

trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 30, model.kernel_size)
E = E / E.max()
t_E_np = np.linspace(0, 30, 500)

def expected_formula(t):
    return np.where(t >= 2.5, (5**2) / (2 * (t**3)), 0)
E_expected_np = expected_formula(t_E_np)
E_expected_np = E_expected_np / E_expected_np.max()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)  
ax1.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
for i in range(c_out.size(1)):
    ax1.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="Tau 5.0",
        color="green",
    )
ax1.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red", linestyle="--"
)
ax1.set_xlim((0, 40))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.legend()
ax1.tick_params(labelbottom=False)
ax2.plot(t_E, E, label="E_Predicted", color="orange")
ax2.set_xlabel('t')
ax2.set_ylabel('E')
ax2.plot(t_E_np, E_expected_np, label="E_Expected", color="purple", linestyle="--")
ax2.set_xlim((0, 40))
ax2.set_ylim((0, 1.1))
ax2.legend()
plt.xlabel("Time")
plt.savefig('Figure_connected_plot_09102024.png', dpi=300)
plt.show()

#Saving of E and time
predicted_E = E
predicted_time = t_E.numpy()               
expected_E = E_expected_np                   
expected_time = t_E_np
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_018_Laminar_Model"
np.save(os.path.join(save_dir, 'E_predicted.npy'), predicted_E)
np.save(os.path.join(save_dir, 't_E_predicted.npy'), predicted_time)
np.save(os.path.join(save_dir, 'E_expected.npy'), expected_E)
np.save(os.path.join(save_dir, 't_E_expected.npy'), expected_time)
print("saved under:", save_dir)
predicted_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_018_Laminar_Model/E_predicted.npy')
predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_018_Laminar_Model/t_E_predicted.npy')
plt.plot(predicted_time, predicted_E, label='E_predicted', color='orange')
plt.plot(expected_time, expected_E, label='E_expected', color='purple', linestyle='--')
plt.xlabel('t')
plt.ylabel('E')
plt.xlim((predicted_time.min(), predicted_time.max()))
plt.ylim((0, 1.1))  
plt.legend()
plt.savefig('E_saved_09102024.png', dpi=300)
plt.show()
