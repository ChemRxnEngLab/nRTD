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
    
base_CNN_dir = os.path.join(base_dir, f'CNN_1st')
#CNN_var_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Disc_variation/Tau_5.0_Laminar_Flow_Model_1st_Layer_Data'
os.makedirs(base_CNN_dir, exist_ok=True)

discretization_confg_second = [(100,49,50),(200,99,100),(300,149,150),(400,199,200),(500,249,250)]
c_conv_results = {}

for n_disc_o, kernel_size, n_disc in discretization_confg_second:
    t_in = torch.linspace(0, 30, n_disc)
    c_in = torch.zeros((1, 1, n_disc))
    c_in[:, :, t_in > 5] = 1
    # Input loading
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
 

    t_conv= torch.tensor(data_time_out[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    c_out = torch.tensor(data_concentration_out[:n_disc_o], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    
    c_out_list = [c_out]
    t_conv_list = [t_conv]
    c_out = torch.cat(c_out_list, dim=0)
    t_conv = torch.cat(t_conv_list, dim=0)
    
    model = RTDModule(
        kernel_size=kernel_size,
        learning_rate=10e-3,
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
    
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=10000,
        deterministic=True,)
    
    trainer.fit(model, dl)
    trainer.test(model, dl)
    c_conv = model(c_in)
    c_conv_results[n_disc_o] = c_conv.detach().numpy()
    
    E = model.net.E[0]
    t_E = torch.linspace(0, 30, model.kernel_size)
    E = E / E.max()
    
    # for disc in discretization_Laminar:
    #         disc_dir = os.path.join(base_CNN_dir, f'disc_{disc}')
    #         os.makedirs(disc_dir, exist_ok=True)
    

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
    
        # Create a unique directory for each discretization level to avoid overwriting
        disc_dir = os.path.join(base_CNN_dir, f'Disc_{disc}')
        os.makedirs(disc_dir, exist_ok=True)
        predicted_E_first = E
        predicted_time_first = t_E.numpy()               
        expected_E_first = E_laminar                   
        expected_time_first = t_l
    
        # Save files in the specific disc directory
        np.save(os.path.join(disc_dir, f'E_predicted_first_layer_{n_disc_o}.npy'), predicted_E_first)
        np.save(os.path.join(disc_dir, f't_E_predicted_first_layer_{n_disc_o}.npy'), predicted_time_first)
        np.save(os.path.join(disc_dir, f'E_expected_first_layer_{n_disc_o}.npy'), expected_E_first)
        np.save(os.path.join(disc_dir, f't_E_expected_first_layer_{n_disc_o}.npy'), expected_time_first)
        np.save(os.path.join(disc_dir, f'c_conv_in_first_layer_{n_disc_o}.npy'), c_conv_results)

        # Plotting and saving the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.plot(t_l, E_laminar, label=f'Tau {tau_l}')
        ax2.plot(t_conv_l, c_out_l, label=f'Tau {tau_l}')
        
        ax1.set_xlim(0, 50)
        ax1.set_xlabel('t')
        ax1.set_ylabel('E')
        ax1.legend()
        
        ax2.plot(t_l, c_0_l, label='c_0', linestyle='--', color='black')
        ax2.set_xlabel('t')
        ax2.set_ylabel('c')
        ax2.legend()
        
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        plt.savefig(os.path.join(disc_dir, f'Laminar_Flow_Model_{current_date}_Disc_{disc}.png'), dpi=300)
        plt.show()
        
        print(f"Discretization: {disc}")
        print(f"Shape of t_conv_l: {t_conv_l.shape}")
        print(f"Shape of c_out_l: {c_out_l.shape}")


discs = [100, 200, 300, 400, 500]  
num_plots = 6
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
tau_l = 5.0
num_plots = 6  # Number of subplots to display

for i, disc in enumerate(discs[:num_plots], start=1):
    # Set up directory and file paths for each disc value
    disc_dir = os.path.join(base_CNN_dir, f'Disc_{disc}')
    t_path = os.path.join(disc_dir, f't_E_expected_first_layer_{discs}.npy')
    t_predicted_path = os.path.join(disc_dir, f't_E_predicted_first_layer_{discs}.npy')
    E_path = os.path.join(disc_dir, f'E_expected_first_layer_{discs}.npy')
    E_predicted_path = os.path.join(disc_dir, f'E_predicted_first_layer_{n_disc_o}.npy')
    
    # Check if files exist
    if os.path.exists(t_path) and os.path.exists(t_predicted_path) and os.path.exists(E_path) and os.path.exists(E_predicted_path):
        # Load data if files exist
        t_expected = np.load(t_path)
        t_predicted = np.load(t_predicted_path)
        E_data = np.load(E_path)
        E_predicted = np.load(E_predicted_path)

        # Determine subplot row and column based on index
        row, col = divmod(i - 1, 2)
        
        # Plot time vs. concentration data
        axs[row, col].plot(t_expected, E_data, label=f'Disc {disc} Expected', color='blue')
        axs[row, col].plot(t_predicted, E_predicted, label=f'Disc {disc} Predicted', color='purple')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('E')
        axs[row, col].legend()
    else:
        # Print message if files are missing
        print(f"Data files not found for Disc {disc} in {disc_dir}. Skipping...")

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()