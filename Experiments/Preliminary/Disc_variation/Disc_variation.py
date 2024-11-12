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

epoch=10000
if wandb.run is not None:
    wandb.finish()
## Data Simulation for the 1st model
base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Disc_variation'
def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E_laminar = np.zeros_like(t)
    E_laminar[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E_laminar

tau_l = 5.0
# discretization_Laminar = [100,200, 334, 500,167,166]
discretization_Laminar = [100,200,300,400,500]

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

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    # ax1.plot(t_l, E_laminar, label=f'Tau {tau_l}')
    # ax2.plot(t_conv_l, c_out_l, label=f'Tau {tau_l}')
    
    # ax1.set_xlim(0,50)
    # ax1.set_xlabel('t')
    # ax1.set_ylabel('E')
    # ax1.legend()
    
    # ax2.plot(t_l, c_0_l, label='c_0', linestyle='--', color='black')
    # ax2.set_xlabel('t')
    # ax2.set_ylabel('c')
    # ax2.legend()
    
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
discretization_confg_second = [(100,49,50),(200,99,100),(300,149,150),(400,199,200),(500,249,250)]
c_conv_results = {}
#c_conv_first = {}

for n_disc_o, kernel_size, n_disc in discretization_confg_second:
    if n_disc_o == 100:
        data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_100_time.npy'))
        data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_100_concentration.npy'))
    elif n_disc_o == 200:
        data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_200_time.npy'))
        data_concentration_out= np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_200_concentration.npy'))
    elif n_disc_o == 300:
        data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_300_time.npy'))
        data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_300_concentration.npy'))
    elif n_disc_o == 400:
        data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_400_time.npy'))
        data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_400_concentration.npy'))
    elif n_disc_o == 500:
        data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_500_time.npy'))
        data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_500_concentration.npy'))
    else:
        raise ValueError(f"Disc do not match: n_disc_o: {n_disc_o}, kernel_size: {kernel_size}, n_disc: {n_disc}")
 # data_time = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{n_disc_o}_time.npy'))
 # data_concentration = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{n_disc_o}_concentration.npy'))
 
    t_conv= torch.tensor(data_time_out[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    c_out = torch.tensor(data_concentration_out[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
   
    t_input = torch.linspace(0, 30, n_disc)
    c_in = torch.zeros((1, 1, n_disc))
    c_in[:, :, t_input > 5] = 1
    
    c_out_list = [c_out]
    t_conv_list = [t_conv]
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
    
    ds = TensorDataset(c_in, c_out)
    dl = DataLoader(ds, batch_size=20, shuffle=True)
    wandb_logger = pl_loggers.WandbLogger(
        project="nRTD",
        log_model=True
    )

    trainer = pl.Trainer(accelerator="auto", max_epochs=epoch, logger=wandb_logger,deterministic=True)
    trainer.fit(model, dl)
    trainer.test(model, dl)
    wandb.finish()
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
   
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.subplots_adjust(hspace=0)
    ax1.plot(t_input, c_in[0, 0, :].numpy(), label="Input Signal", color="blue")
    ax1.plot(t_conv[0, 0, :].numpy(), c_out[0, 0, :].numpy(), label="Expected Output", color="green")
    ax1.plot(t_conv[0, 0, :].numpy(), c_conv[0, 0, :].detach().numpy(), label="Predicted Output", color="red", linestyle="--")
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
   
base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Disc_variation/CNN_1st'
y = [1.17e-6, 9.1083e-7, 4.544e-6, 8.293e-6, 1.8712e-4]
x = [100, 200, 300, 400, 500]
# y = [1.17e-6, 9.1083e-7, 4.544e-6, 8.293e-6]
# x = [100, 200, 300, 400]
discs = [100, 200, 300, 400, 500]  
num_plots = 5

fig, axs = plt.subplots(3, 2, figsize=(12, 10))

for i, disc in enumerate(discs[:num_plots], start=1):
    disc_dir = os.path.join(base_dir, f'Disc_{disc}')
    t_expected_path = os.path.join(disc_dir, f't_E_expected_first_layer_{disc}.npy')
    t_predicted_path = os.path.join(disc_dir, f't_E_predicted_first_layer_{disc}.npy')
    E_expected_path = os.path.join(disc_dir, f'E_expected_first_layer_{disc}.npy')
    E_predicted_path = os.path.join(disc_dir, f'E_predicted_first_layer_{disc}.npy')
    if all(os.path.exists(path) for path in [t_expected_path, t_predicted_path, E_expected_path, E_predicted_path]):
        t_expected = np.load(t_expected_path)
        t_predicted = np.load(t_predicted_path)
        E_expected = np.load(E_expected_path)
        E_predicted = np.load(E_predicted_path)
        row, col = divmod(i - 1, 2)
        
        axs[row, col].plot(t_expected, E_expected, label=fr'$E_{{expected}} \, @n_{{o,1}} = {disc}$', color='blue')
        axs[row, col].plot(t_predicted, E_predicted, label=fr'$E_{{predicted}} \, @n_{{o,1}} = {disc}$', color='purple', linestyle='--')
        #axs[row, col].set_title(f'Disc {disc}')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('E')
        axs[row, col].legend()
        axs[row, col].set_xlim(0, 30)
    else:
        print("Skipping...")

axs[2, 1].loglog(x, y, color='green', marker='o')  # Log scale only on x-axis
axs[2, 1].set_xlabel('Number of Discretization')
axs[2, 1].set_ylabel('Test/Loss')
# axs[2, 1].ticklabel_format(style='sci', axis='y', scilimits=(-8, -8))
# axs[2, 1].set_xticks([100, 200, 300, 400, 500])  
# axs[2, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Format x-axis labels in standard notation
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
axs[2, 1].text(0.03, 0.95, 'Logarithmic Scale',  # Place it near the top
               transform=axs[2, 1].transAxes,  # Use axis-relative coordinates (0 to 1)
               fontsize=8, color='Black', 
               ha='left', va='top', rotation=0)  # No rotation, top alignment
plt.savefig(os.path.join(base_dir, "subplots_disc_variation.png"), dpi=300)
plt.show()


# base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Disc_variation/CNN_1st'
# discs = [100, 200, 300, 400, 500]  
# num_plots = 6  

# fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# for i, disc in enumerate(discs[:num_plots], start=1):
#     disc_dir = os.path.join(base_dir, f'Disc_{disc}')
#     t_expected_path = os.path.join(disc_dir, f't_E_expected_first_layer_{disc}.npy')
#     t_predicted_path = os.path.join(disc_dir, f't_E_predicted_first_layer_{disc}.npy')
#     E_expected_path = os.path.join(disc_dir, f'E_expected_first_layer_{disc}.npy')
#     E_predicted_path = os.path.join(disc_dir, f'E_predicted_first_layer_{disc}.npy')
#     if all(os.path.exists(path) for path in [t_expected_path, t_predicted_path, E_expected_path, E_predicted_path]):
#         t_expected = np.load(t_expected_path)
#         t_predicted = np.load(t_predicted_path)
#         E_expected = np.load(E_expected_path)
#         E_predicted = np.load(E_predicted_path)
    
#         if t_expected.shape != E_expected.shape:
#             print(f"Expected data shape mismatch for Disc {disc}: {t_expected.shape} vs {E_expected.shape}. Skipping...")
#             continue
#         if t_predicted.shape != E_predicted.shape:
#             print(f"Predicted data shape mismatch for Disc {disc}: {t_predicted.shape} vs {E_predicted.shape}. Skipping...")
#             continue
#         row, col = divmod(i - 1, 2)
        
#         axs[row, col].plot(t_expected, E_expected, label=f'Expected (Disc {disc})', color='blue')
#         axs[row, col].plot(t_predicted, E_predicted, label=f'Predicted (Disc {disc})', color='purple', linestyle='--')
#         axs[row, col].set_title(f'Disc {disc}')
#         axs[row, col].set_xlabel('Time')
#         axs[row, col].set_ylabel('E')
#         axs[row, col].legend()
#     else:
#         print(f"One or more data files not found for Disc {disc} in {disc_dir}. Skipping...")

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig(os.path.join(base_dir, "subplots_disc_variation.png"), dpi=300)
# plt.show()


# import matplotlib.pyplot as plt
# import numpy.typing as npt
# import sys
# import os
# module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
# sys.path.append(module_path)
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# import lightning.pytorch as pl
# import numpy as np
# import wandb
# from nRTD import RTDModule
# from lightning.pytorch import loggers as pl_loggers
# import sympy as sp
# import os
# import datetime

# ## Data Simulation for the 1st model
# base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Disc_variation'
# def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
#     E_laminar = np.zeros_like(t)
#     E_laminar[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
#     return E_laminar

# tau_l = 5.0
# discretization_Laminar = [100,200,300,400,500]

# for disc in discretization_Laminar:
#     t_l = np.linspace(0, 60, disc, endpoint=True)  
#     c_0_l = np.zeros_like(t_l)
#     c_0_l[t_l > 5] = 1  
#     E_laminar = laminarflow(t_l, tau_l)
#     E_laminar = E_laminar / E_laminar.max()
#     c_out_l_full = np.convolve(c_0_l, E_laminar / np.sum(E_laminar), mode="full")
#     t_conv_l_full = np.linspace(t_l[0] + t_l[0], t_l[-1] + t_l[-1], len(c_out_l_full))
#     valid_indices = t_conv_l_full <= 60
#     t_conv_l = t_conv_l_full[valid_indices]
#     c_out_l = c_out_l_full[valid_indices]

#     tau_l_dir = os.path.join(base_dir, f'Tau_{tau_l}_Laminar_Flow_Model_1st_Layer_Data')
#     os.makedirs(tau_l_dir, exist_ok=True)

#     np.save(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{disc}_time.npy'), t_conv_l)
#     np.save(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_{disc}_concentration.npy'), c_out_l)

#     # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
#     # ax1.plot(t_l, E_laminar, label=f'Tau {tau_l}')
#     # ax2.plot(t_conv_l, c_out_l, label=f'Tau {tau_l}')
    
#     # ax1.set_xlim(0,50)
#     # ax1.set_xlabel('t')
#     # ax1.set_ylabel('E')
#     # ax1.legend()
    
#     # ax2.plot(t_l, c_0_l, label='c_0', linestyle='--', color='black')
#     # ax2.set_xlabel('t')
#     # ax2.set_ylabel('c')
#     # ax2.legend()
    
#     # current_date = datetime.datetime.now().strftime("%Y%m%d")
#     # plt.savefig(os.path.join(tau_l_dir, f'Laminar_Flow_Model_{current_date}_Disc_{disc}.png'), dpi=300)
#     # plt.show()
#     print(f"Discretization: {disc}")
#     print(f"Shape of t_conv_l: {t_conv_l.shape}")
#     print(f"Shape of c_out_l: {c_out_l.shape}")
    
# base_CNN_dir = os.path.join(base_dir, f'CNN_1st')
# #CNN_var_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Disc_variation/Tau_5.0_Laminar_Flow_Model_1st_Layer_Data'
# os.makedirs(base_CNN_dir, exist_ok=True)

# discretization_confg_second = [(100,49,50),(200,99,100),(300,149,150),(400,199,200),(500,249,250)]
# c_conv_results = {}

# for n_disc_o, kernel_size, n_disc in discretization_confg_second:
#     t_in = torch.linspace(0, 30, n_disc)
#     c_in = torch.zeros((1, 1, n_disc))
#     c_in[:, :, t_in > 5] = 1
#     # Input loading
#     if n_disc_o == 100:
#         data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_100_time.npy'))
#         data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_100_concentration.npy'))
#     elif n_disc_o == 200:
#         data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_200_time.npy'))
#         data_concentration_out= np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_200_concentration.npy'))
#     elif n_disc_o == 300:
#         data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_300_time.npy'))
#         data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_300_concentration.npy'))
#     elif n_disc_o == 400:
#         data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_400_time.npy'))
#         data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_400_concentration.npy'))
#     elif n_disc_o == 500:
#         data_time_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_500_time.npy'))
#         data_concentration_out = np.load(os.path.join(tau_l_dir, f'Tau_{tau_l}_Laminar_Flow_Model_Disc_500_concentration.npy'))
 

#     t_conv= torch.tensor(data_time_out[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#     c_out = torch.tensor(data_concentration_out[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    
#     c_out_list = [c_out]
#     t_conv_list = [t_conv]
#     c_out = torch.cat(c_out_list, dim=0)
#     t_conv = torch.cat(t_conv_list, dim=0)
    
#     model = RTDModule(
#         kernel_size=kernel_size,
#         learning_rate=10e-3,
#         use_scheduler=True,
#         scheduler_kwargs={"factor": 0.5, "patience": 80},
#     )
    
#     c_conv = model(c_in)
#     E = model.net.E[0]
#     t_E = torch.linspace(0, 30, model.kernel_size)
#     E = E / E.max()
#     print(model(c_in).size())
    
#     ds = TensorDataset(c_in, c_out)
#     dl = DataLoader(ds, batch_size=20, shuffle=True)
    
#     trainer = pl.Trainer(
#         accelerator="auto",
#         max_epochs=10,
#         deterministic=True,)
    
#     trainer.fit(model, dl)
#     trainer.test(model, dl)
#     c_conv = model(c_in)
#     c_conv_results[n_disc_o] = c_conv.detach().numpy()
    
#     E = model.net.E[0]
#     t_E = torch.linspace(0, 30, model.kernel_size)
#     E = E / E.max()
    
#     discretization_dir_first_layer = os.path.join(base_CNN_dir, f'Disc_{n_disc_o}')
#     os.makedirs(discretization_dir_first_layer, exist_ok=True)
    
#     predicted_E_first = E
#     predicted_time_first = t_E.numpy()               
#     expected_E_first = E_laminar                   
#     expected_time_first = t_l
    
#     np.save(os.path.join(discretization_dir_first_layer, f'E_predicted_first_layer_{n_disc_o}.npy'), predicted_E_first)
#     np.save(os.path.join(discretization_dir_first_layer, f't_E_predicted_first_layer_{n_disc_o}.npy'), predicted_time_first)
#     np.save(os.path.join(discretization_dir_first_layer, f'E_expected_first_layer_{n_disc_o}.npy'), expected_E_first)
#     np.save(os.path.join(discretization_dir_first_layer, f't_E_expected_first_layer_{n_disc_o}.npy'), expected_time_first)
#     np.save(os.path.join(discretization_dir_first_layer, f'c_conv_in_first_layer_{n_disc_o}.npy'), c_conv_results)
    
#     # Optional: Validate only designated files are present
#     allowed_files = {
#         f't_E_predicted_first_layer_{disc}.npy',
#         f'E_expected_first_layer_{disc}.npy',
#         f't_E_expected_first_layer_{disc}.npy',
#         f'E_predicted_first_layer_{disc}.npy'
#         }
#         # print(f"Discretization: {disc}")
#         # print(f"Shape of t_conv_l: {t_conv_l.shape}")
#         # print(f"Shape of c_out_l: {c_out_l.shape}")

# discs = [100, 200, 300, 400, 500]  
# num_plots = 6
# fig, axs = plt.subplots(3, 2, figsize=(15, 10))
# tau_l = 5.0

# # Loop through each disc value
# for i, disc in enumerate(discs[:num_plots], start=1):
#     disc_dir = os.path.join(base_CNN_dir, f'Disc_{disc}')
    
#     # Paths to expected and predicted files
#     t_path = os.path.join(discretization_dir_first_layer, f't_E_expected_first_layer_{disc}.npy')
#     t_predicted_path = os.path.join(discretization_dir_first_layer, f't_E_predicted_first_layer_{disc}.npy')
#     E_path = os.path.join(discretization_dir_first_layer, f'E_expected_first_layer_{disc}.npy')
#     E_predicted_path = os.path.join(discretization_dir_first_layer, f'E_predicted_first_layer_{disc}.npy')

#     print(f"Checking files for Disc {disc}:")
#     print(f"  t_path exists: {os.path.exists(t_path)}")
#     print(f"  t_predicted_path exists: {os.path.exists(t_predicted_path)}")
#     print(f"  E_path exists: {os.path.exists(E_path)}")
#     print(f"  E_predicted_path exists: {os.path.exists(E_predicted_path)}")

#     # Load and plot data if all files exist
#     if all(os.path.exists(p) for p in [t_path, t_predicted_path, E_path, E_predicted_path]):
#         t_expected = np.load(t_path)
#         t_predicted = np.load(t_predicted_path)
#         E_data = np.load(E_path)
#         E_predicted = np.load(E_predicted_path)
        
#         # Ensure dimensions match for plotting
#         if len(t_expected) != len(E_data) or len(t_predicted) != len(E_predicted):
#             print(f"Skipping Disc {disc} due to mismatched lengths.")
#             continue

#         # Determine subplot position
#         row, col = divmod(i - 1, 2)
        
#         # Plot each disc's expected and predicted data
#         axs[row, col].plot(t_expected, E_data, label=f'Disc {disc} Expected', color='blue')
#         axs[row, col].plot(t_predicted, E_predicted, label=f'Disc {disc} Predicted', color='purple')
#         axs[row, col].set_title(f"Disc {disc}")
#         axs[row, col].set_xlabel('Time')
#         axs[row, col].set_ylabel('E')
#         axs[row, col].legend()
#     else:
#         print(f"Data files not found for Disc {disc} in {disc_dir}. Skipping...")

# # Clean up plot layout
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.suptitle("Expected vs. Predicted for Each Discretization")
# plt.show()