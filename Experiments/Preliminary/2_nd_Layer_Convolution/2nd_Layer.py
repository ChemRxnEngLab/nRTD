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
# discretization_Laminar = [100,200, 334, 500,167,166]
discretization_Laminar = [200, 334]

for disc in discretization_Laminar:
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

    tau_l_dir = os.path.join(base_dir, f'Tau_{tau_l}_Laminar_Flow_Model_1st_Layer_Data')
    os.makedirs(tau_l_dir, exist_ok=True)

    np.save(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{disc}_time.npy'), t_conv_l)
    np.save(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{disc}_concentration.npy'), c_out_l)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(t_l, E_laminar, label=f'Tau {tau_l}')
    ax2.plot(t_conv_l, c_out_l, label=f'Tau {tau_l}')
    
    ax1.set_xlim(0,50)
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

#discretization_confg = [(167,83,83),(200, 99, 100), (334, 166, 167),(100,49,50)]
discretization_confg = [(200, 99, 100), (334, 166, 167)]
c_conv_results = {}
#c_conv_first = {}

for n_disc_o, kernel_size, n_disc in discretization_confg:
    if n_disc_o == 200:
        data_time = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_200_time.npy'))
        data_concentration = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_200_concentration.npy'))
    elif n_disc_o == 334:
        data_time = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_334_time.npy'))
        data_concentration = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_334_concentration.npy'))
    elif n_disc_o == 100:
        data_time = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_100_time.npy'))
        data_concentration = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_100_concentration.npy'))
    elif n_disc_o == 167:
        data_time = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_167_time.npy'))
        data_concentration = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_167_concentration.npy'))
    else:
        raise ValueError(f"Disc do not match: n_disc_o: {n_disc_o}, kernel_size: {kernel_size}, n_disc: {n_disc}")
 # data_time = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{n_disc_o}_time.npy'))
 # data_concentration = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{n_disc_o}_concentration.npy'))
 
    t_conv_tau = torch.tensor(data_time[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    c_out_tau = torch.tensor(data_concentration[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    #print(discretization_number_1st_Layer,"discretization_number_1st_Layer")

    t_input = torch.linspace(0, 30, n_disc)
    c_in = torch.zeros((1, 1, n_disc))
    c_in[:, :, t_input > 5] = 1
    c_out_list = [c_out_tau]
    t_conv_list = [t_conv_tau]
    c_out = torch.cat(c_out_list, dim=0)
    t_conv = torch.cat(t_conv_list, dim=0)

    model = RTDModule(
        kernel_size=kernel_size,
        learning_rate=1e-3,
        use_scheduler=True,
        scheduler_kwargs={"factor": 0.5, "patience": 80},
    )

    c_conv = model(c_in)
    E = model.net.E[0]
    t_E = torch.linspace(0, 30, model.kernel_size)
    E /= E.max()
    print(model(c_in).size())
    
    ds = TensorDataset(c_in, c_out_tau)
    dl = DataLoader(ds, batch_size=20, shuffle=True)

    trainer = pl.Trainer(accelerator="auto", max_epochs=10000, deterministic=True)
    trainer.fit(model, dl)
    trainer.test(model, dl)
    c_conv = model(c_in)
    c_conv_results[n_disc_o] = c_conv.detach().numpy()
    

    E = model.net.E[0]
    t_E = torch.linspace(0, 30, model.kernel_size)
    E = E / E.max()
    E_laminar=E_laminar/ E_laminar.max()
    
    discretization_dir_first_layer = os.path.join(first_layer_CNN_dir, f'Disc_{n_disc_o}')
    os.makedirs(discretization_dir_first_layer, exist_ok=True)
    
    predicted_E_first = E
    predicted_time_first = t_E.numpy()               
    expected_E_first = E_laminar                   
    expected_time_first = t_l
    
    np.save(os.path.join(discretization_dir_first_layer, f'E_predicted_first_layer_{n_disc_o}.npy'), predicted_E_first)
    np.save(os.path.join(discretization_dir_first_layer, f't_E_predicted_first_layer_{n_disc_o}.npy'), predicted_time_first)
    np.save(os.path.join(discretization_dir_first_layer, f'E_expected_first_layer_{n_disc_o}.npy'), expected_E_first)
    np.save(os.path.join(discretization_dir_first_layer, f't_E_expected_first_layer_{n_disc_o}.npy'), expected_time_first)
    np.save(os.path.join(discretization_dir_first_layer, f'c_conv_in_first_layer_{n_disc_o}.npy'), c_conv_results)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.subplots_adjust(hspace=0)
    ax1.plot(t_input, c_in[0, 0, :].numpy(), label="Input Signal", color="blue")
    ax1.plot(t_conv_tau[0, 0, :].numpy(), c_out_tau[0, 0, :].numpy(), label="Expected Output", color="green")
    ax1.plot(t_conv_tau[0, 0, :].numpy(), c_conv[0, 0, :].detach().numpy(), label="Predicted Output", color="red", linestyle="--")
    ax1.set_xlim((0, 30))
    ax1.set_ylim((0, 1.1))
    ax1.set_ylabel('Concentration')
    ax1.legend()

    ax2.plot(t_E, E, label="Predicted E", color="orange")
    ax2.plot(t_l, E_laminar, label="Expected E", color="purple", linestyle="--")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('E')
    ax2.legend()

    plt.savefig(os.path.join(first_layer_CNN_dir, f'Profile_{current_date}_1st_Convolution_Layer_{n_disc_o}.png'), dpi=300)
    plt.show()
    
for disc, conv_result in c_conv_results.items():
    print(f"c_conv_reults_{disc}: {conv_result.shape}")

## Data Simulation for the 2nd model

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

coefficients = {
    'tau_a_val': np.array([1]),   
    'tau_p_val': np.array([2]), 
    'beta_val': np.array([0.1]), 
    'alpha_val': 0.2           
}
discretization_adler = [200, 334]

for disc in discretization_adler:
    t_values_Adler = np.linspace(0, 50, disc, endpoint=True)
    results = compute_inverse_laplace(coefficients, t_values_Adler)
    
    c_0_Adler = c_conv_results[disc]  # Load the relevant c_conv_results for this discretization
    c_0_Adler_normalized = c_0_Adler.flatten()  # Normalize
    
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

        c_out_full_Adler = np.convolve(c_0_Adler_normalized, E_Adler_normalized, mode="full")
        t_conv_full_Adler = np.linspace(t_values_Adler[0] * 2, t_values_Adler[-1] * 2, len(c_out_full_Adler))
        valid_indices = t_conv_full_Adler <= 50 #TIME BE CAREFUL
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
  
## CNN, 2nd layer

adler_CNN_dir = os.path.join(base_dir, f'CNN_2nd')
Adler334_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/2_nd_Layer_Convolution/Adler_tau_a_1_tau_p_2_Disc_334'
Adler167_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/2_nd_Layer_Convolution/Adler_tau_a_1_tau_p_2_Disc_167'
os.makedirs(adler_CNN_dir, exist_ok=True)

#discretization_confg_second = [(200, 132, 334),(100, 65, 167)]
discretization_confg_second = [(200, 133, 334)]
c_conv_results_second = {}

# Ensure you're iterating over the correct config variable
for n_disc, kernel_size, n_disc_o in discretization_confg_second:
    # Input loading
    if n_disc_o == 334:
        data_time_in = np.load(os.path.join(Adler334_dir, 'time_Adler_disc_334.npy'))
        data_concentration_in = np.load(os.path.join(Adler334_dir, 'concentration_Adler_disc_334.npy'))
    elif n_disc_o == 167:
        data_time_in = np.load(os.path.join(Adler167_dir, 'time_Adler_disc_167.npy'))
        data_concentration_in = np.load(os.path.join(Adler167_dir, 'concentration_Adler_disc_167.npy'))
    # elif n_disc_o == 500:
    #     data_time_in = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{500}_time.npy'))
    #     data_concentration_in = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{500}_concentration.npy'))
    # else:
    #     raise ValueError(f"Invalid n_disc_o value: {n_disc_o}")


for n_disc, kernel_size, n_disc_o in discretization_confg_second:
    if n_disc == 200:
        data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{200}_time.npy'))
        data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{200}_concentration.npy'))
    elif n_disc == 100:
        data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{100}_time.npy'))
        data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{100}_concentration.npy'))
    else:
        raise ValueError(f"Invalid n_disc value: {n_disc}")
   # else:
    #     raise ValueError(f"Invalid n_disc value: {n_disc}")

t_conv_adler_second= torch.tensor(data_time_out[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_adler_second = torch.tensor(data_concentration_out[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
t_in_conv_second = torch.tensor(data_time_out[:n_disc], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_in_conv_second = torch.tensor(data_concentration_out[:n_disc], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

c_out_list = [c_out_adler_second]
t_conv_list = [t_conv_adler_second]
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

model = RTDModule(
    kernel_size=132,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

c_conv = model(c_in_conv_second)
E = model.net.E[0]
t_E = torch.linspace(0, 20, model.kernel_size)
E = E / E.max()
print(model(c_in_conv_second).size())

ds = TensorDataset(c_in_conv_second, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000,
    deterministic=True,)

trainer.fit(model, dl)
trainer.test(model, dl)
c_conv = model(c_in_conv_second)
c_conv_results[n_disc_o] = c_conv.detach().numpy()

E = model.net.E[0]
t_E = torch.linspace(0, 20, model.kernel_size)
E = E / E.max()


# predicted_E = E
# predicted_time = t_E
# unified_dir = os.path.expanduser('~/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_20/tCombined_au_a_val_3_tau_p_val_2')
# os.makedirs(unified_dir, exist_ok=True)
# np.save(os.path.join(unified_dir, 'E_predicted_tau_a_val_3.npy'), predicted_E)
# np.save(os.path.join(unified_dir, 't_E_predicted_tau_a_val_3.npy'), predicted_time)


predicted_E_first = E
predicted_time_first = t_E.numpy()               
expected_E_second = E_Adler_normalized                   
expected_time_second = t_conv_Adler
E_Adler=E_Adler/E_Adler.max()

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
ax1.set_xlim((0, 20))
ax1.set_ylim((0, 1.1))
ax1.set_ylabel('Concentration')
ax1.legend()
ax1.tick_params(labelbottom=False)
ax2.plot(t_E, E, label="E", color="orange")
ax2.set_xlabel('t')
ax2.set_ylabel('E')
ax2.plot(t_conv_Adler, E_Adler, label="E (Expected )", color="purple", linestyle="--")
ax2.set_xlim((0, 30))
ax2.set_ylim((0, 1.1))
ax2.legend()
ax2.set_xticks(np.arange(0, 10, 1))  
plt.xlabel("Time")
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_020"
unified_dir = os.path.join(save_dir, f'tau_a_val_{1}_tau_p_val_2_combined')
os.makedirs(unified_dir, exist_ok=True)
plt.savefig(os.path.join(unified_dir, 'Figure_Adler_havarka_Model_tau_a_val_1_tau_p_val_2_combined.png'), dpi=300)
plt.show()
print("Shape of c_conv:", c_conv[0, 0, :].shape)

# save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_020"
# np.save(os.path.join(unified_dir, 'E_predicted_tau_a_val_1_combined.npy'), predicted_E)
# np.save(os.path.join(unified_dir, 't_E_predicted_tau_a_val_1_combined.npy'), predicted_time)
# np.save(os.path.join(unified_dir, 'E_expected_tau_a_val_1_combined.npy'), expected_E)
# np.save(os.path.join(unified_dir, 't_E_expected_tau_a_val_1_combined.npy'), expected_time)
# print("saved under:", unified_dir)
# print("E_predicted",predicted_E)
# print("E_expected",expected_E)
# predicted_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_020/tau_a_val_1_tau_p_val_2_combined/E_predicted_tau_a_val_1_combined.npy')
# predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_020/tau_a_val_1_tau_p_val_2_combined/t_E_predicted_tau_a_val_1_combined.npy')
# plt.plot(predicted_time, predicted_E, label='E_predicted_tau_a_val_1_combined', color='orange')
# plt.plot(expected_time, expected_E, label='E_expected_tau_a_val_1_combined', color='purple', linestyle='--')
# plt.xlabel('t')
# plt.ylabel('E')
# plt.xlim((predicted_time.min(), predicted_time.max()))
# plt.ylim((0, 1.1))  
# plt.xlim(0,25)
# plt.legend()
# plt.xticks(np.arange(0, 10, 1))  
# plt.savefig(os.path.join(unified_dir, 'E_saved_22102024_tau_a_val_1_combined.png'), dpi=300)
# plt.show()

