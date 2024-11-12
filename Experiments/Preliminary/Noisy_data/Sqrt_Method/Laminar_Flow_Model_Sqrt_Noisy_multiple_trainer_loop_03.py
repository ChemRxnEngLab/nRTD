import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")
from nRTD import RTDModule
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import wandb


if wandb.run is not None:
    wandb.finish()
    
epoch=1
n_disc = 100
t_input = torch.linspace(0, 30, n_disc)
c_in = torch.zeros((100, 1, n_disc))
c_in[:, :, t_input > 5] = 1

test_file_numbers = range(30, 100)

def load_data(file_numbers):
    c_out_list = []
    t_conv_list = []
    for file_num in file_numbers:
        
        t_conv_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method/Tau_5.0_Laminar_Flow_Model_Dataset/Tau_5.0_Disc_200_time_Dataset_{file_num:1d}.npy"
        c_out_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method/Tau_5.0_Laminar_Flow_Model_Dataset/Tau_5.0_Disc_200_concentration_noisy_Dataset_{file_num:1d}.npy"
        t_conv = torch.tensor(np.load(t_conv_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        c_out = torch.tensor(np.load(c_out_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_conv_list.append(t_conv)
        c_out_list.append(c_out)
   
    return torch.cat(t_conv_list, dim=0), torch.cat(c_out_list, dim=0)

t_conv_test, c_out_test = load_data(test_file_numbers)
test_ds = TensorDataset(c_in[30:100], c_out_test)  
test_dl = DataLoader(test_ds, batch_size=30, shuffle=False)

train_file_configurations = [30,40,50,60,70,80,90,100]

for num_train_files in train_file_configurations:
    train_file_numbers = range(1, num_train_files + 1)
    t_conv_train, c_out_train = load_data(train_file_numbers)
    train_ds = TensorDataset(c_in[:num_train_files], c_out_train)
    train_dl = DataLoader(train_ds, batch_size=2)
    model = RTDModule(
        kernel_size=99,
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
    E = model.net.E[0] / model.net.E[0].max()
    t_E = torch.linspace(0, 30, model.kernel_size)
    
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
    plt.title(f"Model Behavior with {num_train_files} Training Files")
    plt.show()
    wandb.finish()
