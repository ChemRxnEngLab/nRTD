import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import numpy.typing as npt
import sys
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
from nRTD import RTDModule
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import wandb
from lightning.pytorch import loggers as pl_loggers
import sympy as sp
import os


tau_l = 5.0
discretization_Laminar = [200]
noise_level = 0.001
num_datasets = 100

base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method'
def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E_laminar = np.zeros_like(t)
    E_laminar[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E_laminar

for i in range(num_datasets):
    disc = discretization_Laminar[0]
    t_l = np.linspace(0, 100, disc, endpoint=True)
    c_0_l = np.zeros_like(t_l)
    c_0_l[t_l > 5] = 1

    E_laminar = laminarflow(t_l, tau_l)
    E_laminar /= E_laminar.max()
    c_out_l_full = np.convolve(c_0_l, E_laminar / np.sum(E_laminar), mode="full")
    t_conv_l_full = np.linspace(t_l[0] + t_l[0], t_l[-1] + t_l[-1], len(c_out_l_full))
    valid_indices = t_conv_l_full <= 100
    t_conv_l = t_conv_l_full[valid_indices]
    c_out_l = c_out_l_full[valid_indices]

    variance_scale = noise_level * np.var(c_out_l)
    noise = np.random.normal(0, np.sqrt(variance_scale), c_out_l.shape)
    c_out_l_noisy = c_out_l + noise
    
    tau_l_dir = os.path.join(base_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Dataset')
    os.makedirs(tau_l_dir, exist_ok=True)
    np.save(os.path.join(tau_l_dir, f'Tau_{tau_l}_Disc_{disc}_time_Dataset_{i+1}.npy'), t_conv_l)
    np.save(os.path.join(tau_l_dir, f'Tau_{tau_l}_Disc_{disc}_concentration_noisy_Dataset_{i+1}.npy'), c_out_l_noisy)
    
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    # ax1.plot(t_l, E_laminar, label=f'Tau {tau_l}')
    # ax1.set_xlim(0, 50)
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('E')
    # ax1.legend()
    # ax2.plot(t_conv_l, c_out_l, label='Noiseless Output', color='blue')
    # ax2.plot(t_conv_l, c_out_l_noisy, label='Noisy Output', color='red', linestyle='--')
    # ax2.plot(t_l, c_0_l, label='c_0', linestyle='--', color='black')
    # ax2.set_xlabel('Time')
    # ax2.set_ylabel('Concentration')
    # ax2.legend()
    # current_date = datetime.datetime.now().strftime("%Y%m%d")
    # plt.savefig(os.path.join(tau_l_dir, f'Laminar_Flow_Model_{current_date}_Dataset_{i+1}.png'), dpi=300)
    # plt.show()

## CNN of 1st model
first_layer_CNN_dir = os.path.join(base_dir, 'CNN_1st')
os.makedirs(first_layer_CNN_dir, exist_ok=True)

discretization_confg = [[200,99,100]]
c_conv_results = {}
#c_conv_first = {}

for n_disc_o, kernel_size, n_disc in discretization_confg:
    if n_disc_o == 200:
        data_time = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Disc_200_time_Dataset_1.npy'))
        data_concentration = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Disc_200_concentration_noisy_Dataset_1.npy'))
    else:
        raise ValueError(f"Disc do not match: n_disc_o: {n_disc_o}, kernel_size: {kernel_size}, n_disc: {n_disc}")

    t_conv_tau = torch.tensor(data_time[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    c_out_tau = torch.tensor(data_concentration[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    #print(discretization_number_1st_Layer,"discretization_number_1st_Layer")

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
    # c_conv_first[n_disc_o] = c_conv.detach().numpy()

    # np.save(os.path.join(discretization_dir_first_layer, f'E_predicted_first_layer_{n_disc_o}_{Dataset}.npy'), predicted_E_first)
    # np.save(os.path.join(discretization_dir_first_layer, f't_E_predicted_first_layer_{n_disc_o}_{Dataset}.npy'), predicted_time_first)
    # np.save(os.path.join(discretization_dir_first_layer, f'E_expected_first_layer_{n_disc_o}_{Dataset}.npy'), expected_E_first)
    # np.save(os.path.join(discretization_dir_first_layer, f't_E_expected_first_layer_{n_disc_o}_{Dataset}.npy'), expected_time_first)
    # np.save(os.path.join(discretization_dir_first_layer, f'c_conv_in_first_layer_{n_disc_o_}{Dataset}.npy'), c_conv_results)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.subplots_adjust(hspace=0)
    ax1.plot(t_input, c_in[0, 0, :].numpy(), label="Input Signal", color="blue")
    ax1.plot(t_l, c_out_l_noisy, label="Expected Output", color="green")
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
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    plt.savefig(os.path.join(first_layer_CNN_dir, f'Profile_{current_date}_1st_Convolution_Layer_{n_disc_o}_sqrdnoicsy.png'), dpi=300)
    plt.show()
    
    