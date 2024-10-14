import sys
import os
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
from nrtd import RTDModule
from lightning.pytorch import loggers as pl_loggers

tau_5_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Unified_time_delay'

t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'time_J2_tau3_alpha0.2.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'concentration_J2_tau3_alpha0.2.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

n_disc = 99
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

model = RTDModule(
    kernel_size=100,
    learning_rate=10e-3,
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
plt.plot(t_E, E, label="CNN E Signal", color="orange")
plt.legend()
plt.xlim((0, 40))
plt.ylim((0, 1.1))
plt.show()

print(model(c_in).size())

ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000,
    deterministic=True
)
trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 30, model.kernel_size)
E = E / E.max()

t_plot = torch.linspace(0, 40, 1000).numpy()

#####2 1######

# E_expected = (3.24642 * np.exp(-2.34833 * t_plot) * t_plot #2 1
#                + 0.0392897 * np.exp(-0.851669 * t_plot) * t_plot
#                - 0.477252 * np.exp(-2.34833 * t_plot)
#                + 0.477252 * np.exp(-0.851669 * t_plot))

#####2 3######

E_expected=(
      0.360714 * np.exp(-0.782777 * t_plot) * t_plot 
      + 0.00436552 * np.exp(-0.28389 * t_plot) * t_plot
      - 0.159084 * np.exp(-0.782777 * t_plot)
      + 0.159084 * np.exp(-0.28389 * t_plot)
  ) 
 
 #####5 1#####
 
# E_expected=(123.224 * np.exp(-5.24709 * t_plot) * t_plot**4 
#         + 5.66367e-9 * np.exp(-0.952909 * t_plot) * t_plot**4
#         - 6.36342 * np.exp(-5.24709 * t_plot) * t_plot**3
#         + 8.68252e-6 * np.exp(-0.952909 * t_plot) * t_plot**3
#         - 4.34703 * np.exp(-5.24709 * t_plot) * t_plot**2
#         + 0.00108703 * np.exp(-0.952909 * t_plot) * t_plot**2
#         - 1.97921 * np.exp(-5.24709 * t_plot) * t_plot
#         + 0.0449 * np.exp(-0.952909 * t_plot) * t_plot
#         - 0.450448 * np.exp(-5.24709 * t_plot)
#         + 0.450448 * np.exp(-0.952909 * t_plot))

#####5 3#####

# E_expected = (0.507094 * np.exp(-1.74903 * t_plot) * t_plot**4
#               + 9.81543e-11 * np.exp(-0.317636 * t_plot) * t_plot**4
#               - 0.0785608 * np.exp(-1.74903 * t_plot) * t_plot**3
#               + 1.0707e-7 * np.exp(-0.317636 * t_plot) * t_plot**3
#               - 0.161001 * np.exp(-1.74903 * t_plot) * t_plot**2
#               + 4.02603e-5 * np.exp(-0.317636 * t_plot) * t_plot**2
#               - 0.219912 * np.exp(-1.74903 * t_plot) * t_plot
#               + 0.00498889 * np.exp(-0.317636 * t_plot) * t_plot
#               - 0.150149 * np.exp(-1.74903 * t_plot)
#               + 0.150149 * np.exp(-0.317636 * t_plot))

# E_expected[t_plot < 5] = 0
E_expected =E_expected /E_expected.max()


plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)  
ax1.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    ax1.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="J2_tau3",
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
ax2.plot(t_E, E, label="E", color="orange")
ax2.set_xlabel('t')
ax2.set_ylabel('E')
ax2.plot(t_plot, E_expected, label="E (Expected )", color="purple", linestyle="--")
ax2.set_xlim((0, 150))
ax2.set_ylim((0, 1.1))
ax2.legend()
ax2.set_xticks(np.arange(0, 10, 1))  
plt.xlabel("Time")
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_019_unified_time_delay_Model"
unified_dir = os.path.join(save_dir, f'J_{2}_tau_{3}')
os.makedirs(unified_dir, exist_ok=True)
plt.savefig(os.path.join(unified_dir, 'Figure_Adler_havarka_Model_J_2_tau_3.png'), dpi=300)
plt.show()

predicted_E = E
predicted_time = t_E              
expected_E = E_expected                 
expected_time = t_plot
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_019_unified_time_delay_Model"
np.save(os.path.join(unified_dir, 'E_predicted_J_2_tau_3.npy'), predicted_E)
np.save(os.path.join(unified_dir, 't_E_predicted_J_2_tau_3.npy'), predicted_time)
np.save(os.path.join(unified_dir, 'E_expected_tau_J_2_tau_3.npy'), expected_E)
np.save(os.path.join(unified_dir, 't_E_expected_J_2_tau_3.npy'), expected_time)
print("saved under:", unified_dir)
print("E_predicted",predicted_E)
print("E_expected",expected_E)
predicted_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_019_unified_time_delay_Model/J_2_tau_3/E_predicted_J_2_tau_3.npy')
predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_019_unified_time_delay_Model/J_2_tau_3/t_E_predicted_J_2_tau_3.npy')
plt.plot(predicted_time, predicted_E, label='E_predicted_J_5_tau_3', color='orange')
plt.plot(expected_time, expected_E, label='E_expected_J_5_tau_3', color='purple', linestyle='--')
plt.xlabel('t')
plt.ylabel('E')
plt.xlim((predicted_time.min(), predicted_time.max()))
plt.ylim((0, 1.1))  
plt.xlim(0,25)
plt.legend()
plt.xticks(np.arange(0, 10, 1))  
plt.savefig(os.path.join(unified_dir, 'E_saved_13102024_J_2_tau_3.png'), dpi=300)
plt.show()




