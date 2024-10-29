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

## Data Simulation for the 1st model
base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/2_nd_Layer_Convolution'
def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E_laminar = np.zeros_like(t)
    E_laminar[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E_laminar

tau_l = 5.0
discretization = [200, 334]

for disc in discretization:
    t_l = np.linspace(0, 60, disc, endpoint=True)  
    c_0_l = np.zeros_like(t_l)
    c_0_l[t_l > 5] = 1  
    E_laminar = laminarflow(t_l, tau_l)
    E_laminar = E_laminar / E_laminar.max()
    c_out_l_full = np.convolve(c_0_l, E_laminar / np.sum(E_laminar), mode="full")
    t_conv_l_full = np.linspace(t_l[0] + t_l[0], t_l[-1] + t_l[-1], len(c_out_l_full))
    
    valid_indices = t_conv_l_full <= 60
    t_conv_l = t_conv_l_full[valid_indices]
    c_out_l = c_out_l_full[valid_indices]

    #t_new = t_conv_l  # t_conv_l already matches the time grid

    # Save data
    tau_l_dir = os.path.join(base_dir, f'Tau_{tau_l}_Laminar_Flow_Model_1st_Layer_Data')
    os.makedirs(tau_l_dir, exist_ok=True)

    np.save(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{disc}_time.npy'), t_conv_l)
    np.save(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{disc}_concentration.npy'), c_out_l)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(t_l, E_laminar, label=f'Tau {tau_l}')
    ax2.plot(t_conv_l, c_out_l, label=f'Tau {tau_l}')

    ax1.set_xlim((t_l.min(), t_l.max()))
    ax1.set_xlabel('t')
    ax1.set_ylabel('E')
    ax1.legend()

    ax2.plot(t_l, c_0_l, label='c_0', linestyle='--', color='black')
    ax2.set_xlabel('t')
    ax2.set_ylabel('c')
    ax2.legend()

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    plt.savefig(os.path.join(tau_l_dir, f'Laminar_Flow_Model_{current_date}_Disc_{disc}.png'), dpi=300)
    plt.show()
    print(f"Discretization: {disc}")
    print(f"Shape of t_conv_l: {t_conv_l.shape}")
    print(f"Shape of c_out_l: {c_out_l.shape}")


## CNN of 1st model
first_layer_CNN_dir = os.path.join(base_dir, 'CNN_1st')
os.makedirs(first_layer_CNN_dir, exist_ok=True)
discretization_first_layer = [(200, 99, 100), (334, 166, 167)]
for n_disc_o, kernel_size, n_disc in discretization_first_layer:
    print(f"n_disc_o: {n_disc_o}, kernel_size: {kernel_size}, n_disc: {n_disc}")
    t_conv_tau = torch.tensor(t_conv_l[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    c_out_tau = torch.tensor(c_out_l[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    discretization_number_1st_Layer = len(t_conv_tau)
    print(discretization_number_1st_Layer,"discretization_number_1st_Layer")
    
    
    #Step Function for the 1st model
    t_input = torch.linspace(0, 30, n_disc)
    c_in = torch.zeros((1, 1, n_disc))
    c_in[::2, :, t_input > 5] = 1
    c_in[1::2, :, t_input < 5] = 1
    c_out_list = [c_out_tau]
    t_conv_list = [t_conv_tau]
    c_out = torch.cat(c_out_list, dim=0)
    t_conv = torch.cat(t_conv_list, dim=0)
    
    model = RTDModule(
        kernel_size=kernel_size,
        learning_rate=10e-4,
        use_scheduler=True,
        scheduler_kwargs={"factor": 0.5, "patience": 80},
        )
    c_conv = model(c_in)
    E = model.net.E[0]
    t_E = torch.linspace(0, 30, model.kernel_size)
    E = E / E.max()
    
    print(model(c_in).size())
    ds = TensorDataset(c_in, c_out)
    dl = DataLoader(ds, batch_size=20, shuffle=True)
# wandb_logger = pl_loggers.WandbLogger(
#     project="nRTD",
#     log_model=True
# )

# trainer = pl.Trainer(
#     accelerator="auto",
#     max_epochs=1,
#     logger=wandb_logger,
#     deterministic=True,
# )

    trainer = pl.Trainer(accelerator="auto",max_epochs=10000,deterministic=True)
    trainer.fit(model, dl)
    trainer.test(model, dl)
    c_conv = model(c_in)
    
    E = model.net.E[0]
    t_E = torch.linspace(0, 30, model.kernel_size)
    E = E / E.max()
    E_laminar=E_laminar/ E_laminar.max()
    
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
    ax2.plot(t_l, E_laminar, label="E_Expected", color="purple", linestyle="--")
    ax2.set_xlim(t_l.min(), t_l.max())
    ax2.legend()
    plt.xlabel("Time")
    plt.savefig(os.path.join(first_layer_CNN_dir,f'Profile_{current_date}_1st_Convolution_Layer_{discretization_number_1st_Layer}.png'), dpi=300)
    plt.show()

plt.plot(t_E, E, label="E_Predicted", color="orange")
plt.plot(t_l, E_laminar, label="E_Expected", color="purple", linestyle="--")
plt.xlabel('t')
plt.ylabel('E')
#plt.xlim((t_l.min(), t_l.max()))
plt.xlim((0, 20))  
plt.ylim((0, 1.1))  
plt.legend()
plt.savefig(os.path.join(first_layer_CNN_dir,f'E_{current_date}_1st_Convolution_Layer_{discretization_number_1st_Layer}.png'), dpi=300)
plt.show()

predicted_E_first = E
predicted_time_first = t_E.numpy()               
expected_E_first = E_laminar                   
expected_time_first = t_l
c_conv_in_first=c_conv.detach().numpy()
#base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/2_nd_Layer_Convolution'
#os.makedirs(base_dir, exist_ok=True)
np.save(os.path.join(first_layer_CNN_dir, 'E_predicted_first_layer.npy'), predicted_E_first)
np.save(os.path.join(first_layer_CNN_dir, 't_E_predicted_first_layer.npy'), predicted_time_first)
np.save(os.path.join(first_layer_CNN_dir, 'E_expected_first_layer.npy'), expected_E_first)
np.save(os.path.join(first_layer_CNN_dir, 't_E_expected_first_layer.npy'), expected_time_first)
np.save(os.path.join(first_layer_CNN_dir, 'c_conv_in_first_layer.npy'),c_conv_in_first )
print("saved under:", first_layer_CNN_dir)
# predicted_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/2_nd_Layer_Convolution/E_predicted_combined.npy')
# predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/2_nd_Layer_Convolution/t_E_predicted_combined.npy')
plt.plot(predicted_time_first, predicted_E_first, label='E_predicted', color='orange')
plt.plot(expected_time_first, expected_E_first, label='E_expected', color='purple', linestyle='--')
plt.xlabel('t')
plt.ylabel('E')
plt.xlim((predicted_time_first.min(), predicted_time_first.max()))
plt.ylim((0, 1.1))  
plt.legend()
current_date = datetime.datetime.now().strftime("%Y%m%d")
#plt.savefig(os.path.join(first_layer_CNN_dir,f'E_{current_date}_1st_Convolution_Layer_{discretization_number_1st_Layer}.png'), dpi=300)
plt.show()
print(f"c_conv_in_combined shape: {c_conv_in_first.shape}")

## Data Simulation for the 2nd model

def compute_inverse_laplace(coefficients, t_values):
    s, t = sp.symbols('s t')
    alpha = coefficients['alpha_val']
    results = []  
    for tau_a_val in coefficients['tau_a_val']:  # Loop for each combination
        for tau_p_val in coefficients['tau_p_val']:
            for beta_val in coefficients['beta_val']:
                tau_m_val = (beta_val * (1 - alpha)) / alpha
                F_s = (sp.exp(-tau_p_val * s)) / (1 + beta_val + tau_a_val * s - (beta_val / (1 + tau_m_val * s)))
                print(f"E(s)")
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
                print(f"Inverse Laplace transform E(t):")
                sp.pprint(f_t)
                integral_E_t = np.trapz(E_t, t_values)
                print(f"Integral of E(t) over the time range: {integral_E_t}")
                
    return results
coefficients = {
    'tau_a_val': np.array([1]),   
    'tau_p_val': np.array([2]), 
    'beta_val': np.array([0.1]), 
    'alpha_val': 0.2           
}
t_values_Adler = np.linspace(0, 50, 200, endpoint=True)
discretization_number_Adler_Data = len(t_values_Adler)
results = compute_inverse_laplace(coefficients, t_values_Adler)
c_0_Adler = c_conv_in_first
c_0_Adler_normalized = c_0_Adler.flatten()  # reshape c_0 to a 1D array
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
for result in results:
    tau_a_val = result['tau_a_val']
    tau_p_val = result['tau_p_val']
    tau_m_val = result['tau_m_val']
    beta_val = result['beta_val']
    E_Adler= result['E_t']
    E_Adler_normalized = E_Adler / np.sum(E_Adler)
    c_out_full_Adler = np.convolve(c_0_Adler_normalized, E_Adler_normalized, mode="full")
    t_conv_full_Adler = np.linspace(t_values_Adler[0] + t_values_Adler[0], t_values_Adler[-1] + t_values_Adler[-1], len(c_out_full_Adler))
    valid_indices = t_conv_full_Adler <= 50
    t_conv_Adler = t_conv_full_Adler[valid_indices]
    c_out_Adler = c_out_full_Adler[valid_indices]
    
    # --tau_l_dir = os.path.join(base_dir, f'tau_combined_{tau_l}') # Folder dir
    # os.makedirs(tau_l_dir, exist_ok=True)
    # np.save(os.path.join(tau_l_dir, 'time_combined.npy'), t_conv_l)
    # np.save(os.path.join(tau_l_dir, 'concentration_combined.npy'), c_out_l)    
    adler_dir = os.path.join(base_dir, f'Adler_tau_a_val_{tau_a_val}_tau_p_val_{tau_p_val}_2nd_Layer_Data')
    os.makedirs(adler_dir, exist_ok=True)
    np.save(os.path.join(adler_dir, f'Adler_tau_a_val_{tau_a_val}_tau_p_val_{tau_p_val}_2nd_Layer_simulated_data_time.npy'), t_conv)
    np.save(os.path.join(adler_dir, f'Adler_tau_a_val_{tau_a_val}_tau_p_val_{tau_p_val}_2nd_Layer_simulated_data_concentration.npy'), c_out)
    ax1.plot(t_values_Adler, E_Adler_normalized, label=f'tau_a={tau_a_val}, tau_p={tau_p_val}, tau_m={tau_m_val}, beta={beta_val}')
    ax2.plot(t_conv_Adler, c_out_Adler, label=f'tau_a={tau_a_val}, tau_p={tau_p_val}, beta={beta_val}')
ax1.set_xlabel('t')
ax1.set_ylabel('E(t)')
ax1.legend()
ax2.set_xticks(np.arange(0, 121, 1))
ax2.plot(t_values_Adler, c_0_Adler_normalized, label='c_0', linestyle='--', color='black')
ax2.set_xlim(0, 120)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('t')
ax2.set_ylabel('C')
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(tau_l_dir, f'Adler_Model_{current_date}_Disc_{discretization_number_Adler_Data}.png'), dpi=300)
plt.show()
#initial_base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/2_nd_Layer_Convolution'
for result in results:
    tau_a_val = result['tau_a_val']
    tau_p_val = result['tau_p_val']
    tau_m_val = result['tau_m_val']
    beta_val = result['beta_val']
    #base_dir = os.path.join(initial_base_dir, f'Combined_tau_a_val_{tau_a_val}_tau_p_val_{tau_p_val}_tau_m_val_{tau_m_val}_beta_val_{beta_val}')
    # t_conv = np.load(os.path.join(adler_dir, f'Adler_tau_a_val_{tau_a_val}_tau_p_val_{tau_p_val}_2nd_Layer_simulated_data_time.npy'))
    # c_out = np.load(os.path.join(adler_dir, f'Adler_tau_a_val_{tau_a_val}_tau_p_val_{tau_p_val}_2nd_Layer_simulated_data_concentration.npy'))
    
## CNN, 2nd layer

adler_CNN_dir = os.path.join(base_dir, f'CNN_2nd')
os.makedirs(adler_CNN_dir, exist_ok=True)

        # t_conv_tau = torch.tensor(t_conv_l, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # c_out_tau = torch.tensor(c_out_l, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# assert os.path.exists(os.path.join(adler_CNN_dir, 'time.npy'))
# assert os.path.exists(os.path.join(adler_CNN_dir, 'concentration.npy'))
# assert os.path.exists(os.path.join(base_dir, 't_E_predicted.npy'))
# assert os.path.exists(os.path.join(base_dir, 'c_conv_in_100.npy'))
# print("All files found successfully.")

t_conv_adler_second = torch.tensor(t_conv_Adler, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_adler_second = torch.tensor(c_out_Adler, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
t_in_conv_second= torch.tensor(predicted_time_first, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#t_in_conv_second= torch.linspace(0, 30, 200).unsqueeze(0).unsqueeze(0)
c_in_conv_second = torch.tensor(c_conv_in_first, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
discretization_number_2nd_Layer = len(t_conv_tau)
# print(discretization_number_1st_Layer,"discretization_number_1st_Layer")
# print(f"t_conv_tau shape: {t_conv_tau.shape}")
# print(f"c_out_tau shape: {c_out_tau.shape}")
# print(f"t_in_conv shape: {t_in_conv.shape}")
# print(f"c_in_conv shape: {c_in_conv.shape}")

c_out_list = [c_out_tau]
t_conv_list = [t_conv_tau]
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

model = RTDModule(
    kernel_size=131,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

c_conv = model(c_in_conv_second)
E = model.net.E[0]
t_E = torch.linspace(0, 20, model.kernel_size)
E = E / E.max()

# plt.figure()
# plt.plot(t_in_conv.squeeze(0).numpy(), c_in_conv.squeeze(0).squeeze(0).numpy(), label="SF", color="blue")
# for i in range(c_out.size(1)):
#     plt.plot(t_conv[0, i, :].numpy(), c_out[0, i, :].numpy(), label="tau_a_val_1_tau_p_val_2", color="green")
# min_length = min(t_conv.shape[-1], c_conv.shape[-1])
# plt.plot(t_conv[0, 0, :min_length].numpy(), c_conv[0, 0, :min_length].detach().numpy(), label="Predicted", color="red")
# plt.plot(t_E, E, label="E", color="orange")
# plt.legend()
# plt.xlim((0, 40))
# plt.ylim((0, 1.1))
# plt.show()

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
# fig.subplots_adjust(hspace=0)  

# ax1.plot(t_in_conv.squeeze().numpy(), c_in_conv[0, :].numpy(), label="SF", color="blue")
# for i in range(c_out.size(1)):
#     ax1.plot(t_conv[0, i, :].numpy(), c_out[0, i, :].numpy(), label="tau_a_val_3_tau_p_val_2", color="green")
# ax1.plot(
#     t_conv[0, 0, :min_length].numpy(),
#     c_conv[0, 0, :min_length].detach().numpy(),
#     label="Predicted",
#     color="red",
#     linestyle="--"
# )
# ax1.set_xlim((0, 150))
# ax1.set_ylim((0, 1.1))
# ax1.set_ylabel('Concentration')
# ax1.legend()
# ax1.tick_params(labelbottom=False)

# ax2.plot(t_E, E, label="E", color="orange")
# ax2.set_xlabel('t')
# ax2.set_xlim((0, 40))
# ax2.set_ylim((0, 1.1))
# ax2.set_ylabel('E')
# ax2.legend()
# plt.show()

print(model(c_conv_in_first).size())

ds = TensorDataset(c_conv_in_first, c_out_adler_second)
dl = DataLoader(ds, batch_size=20, shuffle=True)
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000,
    deterministic=True,)

trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_conv_in_first)
E = model.net.E[0]
t_E = torch.linspace(0, 20, model.kernel_size)
E = E / E.max()
t_E_np = np.linspace(0, 20, 500)

# predicted_E = E
# predicted_time = t_E
# unified_dir = os.path.expanduser('~/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_20/tCombined_au_a_val_3_tau_p_val_2')
# os.makedirs(unified_dir, exist_ok=True)
# np.save(os.path.join(unified_dir, 'E_predicted_tau_a_val_3.npy'), predicted_E)
# np.save(os.path.join(unified_dir, 't_E_predicted_tau_a_val_3.npy'), predicted_time)

def compute_inverse_laplace(coefficients, t_values):
    s, t = sp.symbols('s t', real=True, positive=True)
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

coefficients = {
    'tau_a_val': np.array([1]),
    'tau_p_val': np.array([2]),
    'beta_val': np.array([0.1]),
    'alpha_val': 0.2
}
t_plot = np.linspace(0, 20, )
inverse_laplace_results = compute_inverse_laplace(coefficients, t_plot)
E_expected = inverse_laplace_results[0]['E_t']
E_expected = E_expected / E_expected.max()
# np.save(os.path.join(unified_dir, 'E_expected_tau_a_val_3.npy'), E_expected)
# np.save(os.path.join(unified_dir, 't_E_expected_tau_a_val_3.npy'), t_plot)

plt.figure()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.subplots_adjust(hspace=0)  
ax1.plot(t_in_conv_second.squeeze().numpy(), c_in_conv_second.squeeze().squeeze().numpy(), label="SF", color="blue")
for i in range(c_out.size(1)):
    ax1.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="tau_a_val_1_tau_p_val_2",
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
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_020"
unified_dir = os.path.join(save_dir, f'tau_a_val_{1}_tau_p_val_2_combined')
os.makedirs(unified_dir, exist_ok=True)
plt.savefig(os.path.join(unified_dir, 'Figure_Adler_havarka_Model_tau_a_val_1_tau_p_val_2_combined.png'), dpi=300)
plt.show()

predicted_E = E
predicted_time = t_E              
expected_E = E_expected                 
expected_time = t_plot
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_020"
np.save(os.path.join(unified_dir, 'E_predicted_tau_a_val_1_combined.npy'), predicted_E)
np.save(os.path.join(unified_dir, 't_E_predicted_tau_a_val_1_combined.npy'), predicted_time)
np.save(os.path.join(unified_dir, 'E_expected_tau_a_val_1_combined.npy'), expected_E)
np.save(os.path.join(unified_dir, 't_E_expected_tau_a_val_1_combined.npy'), expected_time)
print("saved under:", unified_dir)
print("E_predicted",predicted_E)
print("E_expected",expected_E)
predicted_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_020/tau_a_val_1_tau_p_val_2_combined/E_predicted_tau_a_val_1_combined.npy')
predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_020/tau_a_val_1_tau_p_val_2_combined/t_E_predicted_tau_a_val_1_combined.npy')
plt.plot(predicted_time, predicted_E, label='E_predicted_tau_a_val_1_combined', color='orange')
plt.plot(expected_time, expected_E, label='E_expected_tau_a_val_1_combined', color='purple', linestyle='--')
plt.xlabel('t')
plt.ylabel('E')
plt.xlim((predicted_time.min(), predicted_time.max()))
plt.ylim((0, 1.1))  
plt.xlim(0,25)
plt.legend()
plt.xticks(np.arange(0, 10, 1))  
plt.savefig(os.path.join(unified_dir, 'E_saved_22102024_tau_a_val_1_combined.png'), dpi=300)
plt.show()

