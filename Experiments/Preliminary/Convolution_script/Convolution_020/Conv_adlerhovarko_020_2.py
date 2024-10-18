import sys
import os
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import wandb
from nrtd import RTDModule
# if wandb.run is not None:
#     wandb.finish()
# module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
# sys.path.append(module_path)

laminar_model_dir='/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_018_Laminar_Model/Laminar Flow'
adler_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Adler_havarka_Model/Combined_tau_a_val_3_tau_p_val_2_tau_m_val_0.4000000000000001_beta_val_0.1'

t_conv_tau = torch.tensor(np.load(os.path.join(adler_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(adler_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

t_in_conv = torch.tensor(np.load(os.path.join(laminar_model_dir, 't_E_predicted.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_in_conv = torch.tensor(np.load(os.path.join(laminar_model_dir, 'c_conv_in.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# n_disc = 88
# t_input = torch.linspace(0, 25, n_disc)
# c_in = torch.zeros((1, 1, n_disc))
# c_in[::2, :, t_input > 5] = 1
# c_in[1::2, :, t_input < 5] = 1

#file_numbers = range(1, 21)
c_out_list = []
t_conv_list = []

c_out_list.append(c_out_tau)
t_conv_list.append(t_conv_tau)


# for i, file_num in enumerate(file_numbers):

#     pass


c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

plt.show()
print(f"c_in size: {c_in_conv}")
print(f"c_out size: {c_out.size()}")
print(f"t_conv size: {t_conv.size()}")

model = RTDModule(
    kernel_size=66,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)
c_conv = model(c_in_conv)
E = model.net.E[0]
t_E = torch.linspace(0, 20, model.kernel_size)
E = E /E.max()
j = 0  

plt.figure()
plt.plot(t_in_conv, c_in_conv[j, 0, :].numpy(), label="SF", color="blue")

# for i in range(c_out.size(1)):
#     plt.plot(
#         t_conv[j, i, :].numpy(),
#         c_out[j, i, :].numpy(),
#         label=f"Exp_{file_numbers[i]}" if i < len(file_numbers) else "Tau 1",
#         color="green",
#     )
for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label= "tau_a_val_3_tau_p_val_2",
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

print(model(c_in_conv).size())


ds = TensorDataset(c_in_conv, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)
# wandb_logger = pl_loggers.WandbLogger(
#     project="nRTD",
#     log_model=True
# )

# trainer = pl.Trainer(
#     accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     max_epochs=10000,
#     logger=wandb_logger,
#     deterministic=True,
# )
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000,
    deterministic=True,
)

trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in_conv)
E = model.net.E[0]
t_E = torch.linspace(0, 20, model.kernel_size)
E = E/E.max()

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
    'tau_p_val': np.array([2]),
    'beta_val': np.array([0.1]),
    'alpha_val': 0.2
}
t_plot = np.linspace(0, 20,100)
inverse_laplace_results = compute_inverse_laplace(coefficients, t_plot)
E_expected = inverse_laplace_results[0]['E_t']
E_expected =E_expected /E_expected.max()

# plt.figure()
# plt.plot(t_input, c_in[0, 0, :].numpy(), label="c_in", color="blue")

# for i in range(c_out.size(1)):
#     plt.plot(
#         t_conv[0, i, :].numpy(),
#         c_out[0, i, :].numpy(),
#         label="tau_a_val_1_tau_p_val_2",
#         color="green",
#     )

# plt.plot(
#     t_conv[0, 0, :].numpy(),
#     c_conv[0, 0, :].detach().numpy(),
#     label="c_predicted",
#     color="red",
# )
# plt.plot(t_E, E, label="E", color="orange")
# plt.plot(t_plot, E_expected, label="E_predicted", color="purple", linestyle="--")
# plt.xlim((0, 50))
# plt.ylim((0, 1.1))
# plt.legend()
# ax = plt.gca()  
# ax.set_xticks(np.arange(0, 10, 1))  
# plt.savefig("Figure_Adler_havarka_Model_tau_a_val_1_tau_p_val_2_200disc_001")
# plt.show()
plt.figure()
plt.plot(t_conv_tau, c_in_conv[0, 0, :].numpy(), label="SF", color="blue")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)  
ax1.plot(t_conv_tau, c_in_conv[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    ax1.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="tau_a_val_3_tau_p_val_2",
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
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_20"
unified_dir = os.path.join(save_dir, f'tCombined_au_a_val_{3}_tau_p_val_2')
os.makedirs(unified_dir, exist_ok=True)
plt.savefig(os.path.join(unified_dir, 'Figure_Adler_havarka_Model_tau_a_val_3_tau_p_val_2_combined.png'), dpi=300)
plt.show()

predicted_E = E
predicted_time = t_E              
expected_E = E_expected                 
expected_time = t_plot
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_015_Adler_havarka_Model"
np.save(os.path.join(unified_dir, 'E_predicted_tau_a_val_3.npy'), predicted_E)
np.save(os.path.join(unified_dir, 't_E_predicted_tau_a_val_3.npy'), predicted_time)
np.save(os.path.join(unified_dir, 'E_expected_tau_a_val_3.npy'), expected_E)
np.save(os.path.join(unified_dir, 't_E_expected_tau_a_val_3.npy'), expected_time)
print("saved under:", unified_dir)
print("E_predicted",predicted_E)
print("E_expected",expected_E)
predicted_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_015_Adler_havarka_Model/tau_a_val_3_tau_p_val_2/E_predicted_tau_a_val_3.npy')
predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_015_Adler_havarka_Model/tau_a_val_3_tau_p_val_2/t_E_predicted_tau_a_val_3.npy')
plt.plot(predicted_time, predicted_E, label='E_predicted_tau_a_val_3', color='orange')
plt.plot(expected_time, expected_E, label='E_expected_tau_a_val_3', color='purple', linestyle='--')
plt.xlabel('t')
plt.ylabel('E')
plt.xlim((predicted_time.min(), predicted_time.max()))
plt.ylim((0, 1.1))  
plt.xlim(0,25)
plt.legend()
plt.xticks(np.arange(0, 10, 1))  
plt.savefig(os.path.join(unified_dir, 'Combined_E_saved_10102024_tau_a_val_3.png'), dpi=300)
plt.show()



