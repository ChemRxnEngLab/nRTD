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
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)

# module_path = r"D:\Tuana\nRTD\lib"
# sys.path.append(module_path)

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

# Base and save directories
#base_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model_dynamic/tau_5.0_with_varying_step"
base_dir =r"D:\Tuana\nRTD\Experiments\Preliminary\Litrature\Laminar_Flow_Model_dynamic\tau_5.0_with_varying_step"
#save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_Laminar_Flow_Dynamic"
save_dir=r"D:\Tuana\nRTD\Experiments\Preliminary\Convolution_script\Convolution_Laminar_Flow_Dynamic"
# Parameters
n_variations = 100  # Use first 100 variations for training
c_in_list = []
c_out_list = []

# Load first 100 variations for training
for i in np.linspace(0, 1, n_variations):  # n_variations will control the number of files
    i_str = f"{i:.2f}".replace('.', '_')  # Format i for filenames
    variation_dir = os.path.join(base_dir, f"variation_{i_str}")  # Directory for each variation

    t_conv = np.load(os.path.join(variation_dir, "time.npy"))
    c_out = np.load(os.path.join(variation_dir, "concentration.npy"))
    input_func = np.load(os.path.join(variation_dir, "input_function_interp.npy"))

    c_in_list.append(torch.tensor(input_func, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    c_out_list.append(torch.tensor(c_out, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

# Prepare training data
c_in_train = torch.cat(c_in_list, dim=0)  # Using all 100 for training
c_out_train = torch.cat(c_out_list, dim=0)

# Load test files
test_time_path = os.path.join(base_dir, "time_test.npy")
test_concentration_path = os.path.join(base_dir, "concentration_test.npy")
test_input_function_path = os.path.join(base_dir, "input_test.npy")

# Check if test files exist
if not all(os.path.exists(path) for path in [test_time_path, test_concentration_path, test_input_function_path]):
    raise FileNotFoundError("One or more test file components are missing at the specified path.")

# Load test data
test_t_conv = torch.tensor(np.load(test_time_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
test_c_out = torch.tensor(np.load(test_concentration_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
test_input_func = torch.tensor(np.load(test_input_function_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Create datasets and dataloaders
train_dataset = TensorDataset(c_in_train, c_out_train)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

test_dataset = TensorDataset(test_input_func, test_c_out)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("Training Input Shape (c_in_train):", c_in_train.shape)
print("Training Output Shape (c_out_train):", c_out_train.shape)
print("Testing Input Shape (test_input_func):", test_input_func.shape)
print("Testing Output Shape (test_c_out):", test_c_out.shape)
print(t_conv.shape)
print(c_out.shape)
# Model initialization
model = RTDModule(
    kernel_sizes=[251],
    kernel_times=[(0.0, 150)],
    learning_rate=1e-4,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

# Logger setup
wandb_logger = pl_loggers.WandbLogger(project="nRTD", log_model=True)

# Training the model
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=30000,
    logger=wandb_logger,
    deterministic=True,
)

# Train and test the model
trainer.fit(model, train_loader)
results = trainer.test(model, test_loader)

# Time and expected values
t_e_1 = 150
c_conv = model(c_in_train)
E = model.net.E[0]
t_E = torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
E = E / E.max()

# Expected E calculation
t_E_np = np.linspace(0, 150, 251)

def expected_formula(t):
    return np.where(t >= 2.5, (2.5**2) / (2 * (t**3)), 0)

E_expected = expected_formula(t_E_np)
E_expected /= E_expected.max()

# Plot E Predicted vs. Expected
plt.figure(figsize=(10, 6))

# Convert E_pred to NumPy if it's a PyTorch tensor and remove detach()
if isinstance(E, torch.Tensor):
    E = E.detach().cpu().numpy()

plt.plot(t_E, E, label="E Predicted", color="orange")
plt.plot(t_E_np, E_expected, label="E Expected", linestyle="--", color="purple")
plt.xlabel("Time")
plt.ylabel("E")
plt.legend()
plt.title("E Predicted vs. Expected")
plt.savefig(os.path.join(save_dir, "E_pred_vs_expected.png"), dpi=300)
plt.show()

# Save results
np.save(os.path.join(save_dir, "E_predicted.npy"), E)
np.save(os.path.join(save_dir, "t_E_predicted.npy"), t_E.numpy())
np.save(os.path.join(save_dir, "E_expected.npy"), E_expected)
np.save(os.path.join(save_dir, "t_E_expected.npy"), t_E_np)

index = 50  # Change this index to plot different samples

# Get the data for the selected index
c_in_sample = c_in_train[index].detach().cpu().numpy()  # input function
c_out_sample = c_out_train[index].detach().cpu().numpy()  # actual output
c_conv_sample = c_conv[index].detach().cpu().numpy()  # predicted output

# Time vector for the x-axis (assuming it's the same for all)
time_vector_in =  np.linspace(0, 150, 250) # Time associated with c_in, c_out, c_conv
time_vector_out =  np.linspace(0, 300, 500)
# Plot c_in, c_out, and c_conv for the selected sample
plt.figure(figsize=(10, 6))
plt.plot(time_vector_in, c_in_sample[0], label="c_in (Input)", linestyle='-', color='blue')
plt.plot(time_vector_out, c_out_sample[0], label="c_out (Actual Output)", linestyle='--', color='green')
plt.plot(time_vector_out, c_conv_sample[0], label="c_conv (Predicted Output)", linestyle='-.', color='orange')


print(c_in_sample.shape)
print(c_out_sample.shape)
print(c_conv_sample.shape)
# Print the shapes of key training and testing data
print("Training Input Shape (c_in_train):", c_in_train.shape)
print("Training Output Shape (c_out_train):", c_out_train.shape)
print("Testing Input Shape (test_input_func):", test_input_func.shape)
print("Testing Output Shape (test_c_out):", test_c_out.shape)

# Optional: Print the actual data if needed (for debugging)
print("Sample Training Input (c_in_train[0]):", c_in_train[0])
print("Sample Training Output (c_out_train[0]):", c_out_train[0])
print("Sample Testing Input (test_input_func[0]):", test_input_func[0])
print("Sample Testing Output (test_c_out[0]):", test_c_out[0])




# Labels and title

plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Comparison of c_in, c_out, and c_conv for a Sample")
plt.legend()
plt.grid(True)

# Save the plot to a file (optional)
#plt.savefig(os.path.join(save_dir, "c_in_c_out_c_conv_sample.png"), dpi=300)

# Show the plot
plt.show()
