import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")
from nRTD import RTDModule
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

n_disc = 100
t_input = torch.linspace(0, 30, n_disc)
c_in = torch.zeros((10, 1, n_disc))
c_in[:, :, t_input > 5] = 1
train_file_numbers = range(1, 6)  
test_file_numbers = range(6, 11)

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

t_conv_train, c_out_train = load_data(train_file_numbers)
t_conv_test, c_out_test = load_data(test_file_numbers)
train_ds = TensorDataset(c_in[:5], c_out_train)  # First 5 for training
test_ds = TensorDataset(c_in[5:], c_out_test)    # Last 5 for testing
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=2, shuffle=False)

model = RTDModule(
    kernel_size=99,
    learning_rate=1e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

# Set up the logger
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True)

# Set up PyTorch Lightning Trainer
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10,
    logger=wandb_logger,
    deterministic=True
)

# Train the model
trainer.fit(model, train_dl)

# Test the model
trainer.test(model, test_dl)

# Postprocessing: Visualize the results
c_conv = model(c_in[:5])  # Forward pass on training data
E = model.net.E[0] / model.net.E[0].max()
t_E = torch.linspace(0, 30, model.kernel_size)

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

# Plot training results
for i in range(c_out_train.size(1)):
    plt.plot(t_conv_train[0, i, :].numpy(), c_out_train[0, i, :].numpy(), label=f"Exp_{train_file_numbers[i]}", color="green")

plt.plot(t_conv_train[0, 0, :].numpy(), c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
plt.plot(t_E, E, label="E", color="orange")
plt.xlim((0, 30))
plt.ylim((0, 2))
plt.legend()
plt.savefig("train_test_split_result.png")
plt.show()
