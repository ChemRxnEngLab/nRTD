import matplotlib.pyplot as plt
import numpy.typing as npt
import sys
import os
module_path = os.path.expanduser("lib")
sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import numpy as np
import wandb
# module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
# sys.path.append(module_path)

module_path = r"D:\Tuana\nRTD\lib"
sys.path.append(module_path)

from nRTD.rtd_fitting_2 import RTDModule
from nRTD.rtd_net_4 import RTDNet
from lightning.pytorch import loggers as pl_loggers
import os
import datetime
import sympy as sp
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch


if wandb.run is not None:
    wandb.finish()
#if wandb.run is not None:
#    wandb.finish()
#os.chdir(r"D:\Tuana\nRTD\Experiments\Preliminary\Noisy_data\Sqrt_Method")

#os.chdir("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method")


epoch=17000
t_e_1=50
n_disc = 100
t_input = torch.linspace(0, 50, n_disc)
c_in = torch.zeros((1000, 1, n_disc))
c_in[:, :, t_input > 5] = 1

# test_file ="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model/tau_5.0_disc_200_100s"
# save_dir="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method"

test_file =r"D:\Tuana\nRTD\Experiments\Preliminary\Litrature\Laminar_Flow_Model\tau_5.0_disc_200_100s"
save_dir=r"D:\Tuana\nRTD\Experiments\Preliminary\Noisy_data\Sqrt_Method"

t_conv_path = os.path.join(test_file, "time.npy")
c_out_path = os.path.join(test_file, "concentration.npy")
t_conv_test = torch.tensor(np.load(t_conv_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_test = torch.tensor(np.load(c_out_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
print(t_conv_test)
#test_file_numbers = range(30, 100)

def load_data(file_numbers):
    c_out_list = []
    t_conv_list = []
    dataset_dir = r"D:\Tuana\nRTD\Experiments\Preliminary\Noisy_data\Sqrt_Method\Tau_5.0_Laminar_Flow_Model_Dataset"
    #dataset_dir="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method/Tau_5.0_Laminar_Flow_Model_Dataset"

    for i in file_numbers: 
        t_conv_path = os.path.join(dataset_dir, f"Tau_5.0_Disc_200_time_Dataset_{i}.npy")
        c_out_path = os.path.join(dataset_dir, f"Tau_5.0_Disc_200_concentration_noisy_Dataset_{i}.npy")
    
        
        if not (os.path.exists(t_conv_path) and os.path.exists(c_out_path)):
            print(f"Warning: Files for dataset {i} do not exist.")
            continue

        t_conv = torch.tensor(np.load(t_conv_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        c_out = torch.tensor(np.load(c_out_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_conv_list.append(t_conv)
        c_out_list.append(c_out)


    t_conv_tensor = torch.cat(t_conv_list, dim=0)
    c_out_tensor = torch.cat(c_out_list, dim=0)
    return t_conv_tensor, c_out_tensor
print("c_in",c_in.shape)
print("c_out_test",c_out_test.shape)

c_in_test = c_in[:1, :, :]  
print("After adjustment - c_in_test shape:", c_in_test.shape)
test_ds = TensorDataset(c_in_test, c_out_test)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)


train_file_configurations = [1]

for num_train_files in train_file_configurations:
    train_file_numbers = range(1, num_train_files + 1)
    t_conv_train, c_out_train = load_data(train_file_numbers)
    train_ds = TensorDataset(c_in[:num_train_files], c_out_train)
    train_dl = DataLoader(train_ds, batch_size=num_train_files)

    model = RTDModule(
        kernel_sizes=[101],
        kernel_times=[(0.0, 50)],
        learning_rate=1e-3,
        use_scheduler=True,
        scheduler_kwargs={"factor": 0.5, "patience": 80},
    )
    wandb_logger = pl_loggers.WandbLogger(
        project="nRTD",
        log_model=True,
        name=f"Training_with_{num_train_files}_files",
        reinit=True
    )
    
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=epoch,
        logger=wandb_logger,
        deterministic=True,
        log_every_n_steps=1 
    )
    
    print(f"Training with {num_train_files} training files.")
    trainer.fit(model, train_dl)
    test_results = trainer.test(model, test_dl)

    c_conv = model(c_in[:num_train_files])  
    E = model.net.E[0]
    t_E =torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
    E = E / E.max()
    results_dir = os.path.join(save_dir, "Results_Noisy_Data")
    os.makedirs(results_dir, exist_ok=True)
    t_E_save_path = os.path.join(results_dir, f"t_E_{num_train_files}.npy")
    E_save_path = os.path.join(results_dir, f"E_{num_train_files}.npy")
    c_save_path = os.path.join(results_dir, f"c_conv_{num_train_files}.npy")
    np.save(t_E_save_path, t_E)
    np.save(E_save_path, E)
    np.save(c_save_path, c_conv.detach().numpy())
    plt.figure(figsize=(10, 6))
    plt.plot(t_input, c_in[0, 0, :].numpy(), label="Input Signal (SF)", color="blue")

    for i in range(num_train_files):
        plt.plot(t_conv_train[i, 0, :].numpy(), c_out_train[i, 0, :].numpy(), 
                 label=f"Training Data {i+1}", color="green")
    
    plt.plot(t_conv_train[0, 0, :].numpy(), c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
    plt.plot(t_E, E, label="System Response (E)", color="orange")
    plt.xlim((0, 30))
    plt.ylim((0, 2))
    plt.legend()
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    plt.savefig(os.path.join(save_dir, f"Noisy_data_{current_date}_{num_train_files}.png"), dpi=300)
    plt.show()
    wandb.finish()

# tau_5_dir_lam="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model/tau_5.0_disc_200_100s"
# t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir_lam, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir_lam, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# predicted_c = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/c_conv_in.npy')
# predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/t_E_predicted.npy')
# t_conv_100=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method/Results_Noisy_Data/t_E_100.npy")
# c_conv_100=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method/Results_Noisy_Data/c_conv_100.npy")

# from ICIW_Plots import make_square_subplots
# import matplotlib.pyplot as plt
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import ICIW_Plots.colors as ICIWcolors
# from ICIW_Plots.figures import Elsevier_Sizes
# import datetime
# from sympy import ceiling
# from ICIW_Plots import make_square_ax, cm2inch
# from ICIW_Plots import make_rect_ax

# # Ensure the tensors are squeezed to 1D arrays
# t_conv_tau_1d = t_conv_tau.squeeze().numpy()  # Shape becomes (200,)
# c_out_tau_1d = c_out_tau.squeeze().numpy()   # Shape becomes (200,)
# predicted_c_1d = predicted_c.squeeze()       # Ensure compatibility for numpy array
# c_conv_100_1d = c_conv_100.squeeze()         # Ensure compatibility for numpy array
# c_conv_100_first_config = c_conv_100_1d[0, :]

# plt.style.use("ICIWstyle")



# # Plotting
# fig = plt.figure()
# ax = make_rect_ax(
#     fig,
#     ax_width=7.3 * cm2inch,
#     ax_height=5 * cm2inch,
#     xlabel="$t$ / $s$",
#     ylabel="$x$ / $1$",
# )

# ax.plot(
#     t_conv_tau_1d,  # Ensure this is 1D
#     predicted_c_1d,
#     label=r"$\hat{E}(t)$",
#     color="black",
# )


# # Plot original data
# ax.plot(
#     t_conv_tau_1d,
#     c_out_tau_1d,
#     label=r"$\hat{x}(t)$",
#     color="purple",
#     linestyle="--",
# )


# ax.plot(
#     t_conv_tau_1d,  # Ensure this is also 1D
#     c_conv_100_first_config,
#     label=r"$E(t)$",
#     color=ICIWcolors.KELLYGREEN,
# )

# # Plot predicted E(t)

# # Customize plot
# ax.set_xlim((0, 50))
# ax.set_ylim((-0.1, 1.1))
# ax.legend(loc="best")

# plt.show()
# print(t_conv_tau_1d.shape)
# print(predicted_c_1d.shape)
# print(c_conv_100_1d.shape)