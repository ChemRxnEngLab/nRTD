import matplotlib.pyplot as plt
import numpy.typing as npt
import sys
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import numpy as np
import wandb
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
from nRTD.rtd_fitting_2 import RTDModule
from nRTD.rtd_net_4 import RTDNet
from lightning.pytorch import loggers as pl_loggers

if wandb.run is not None:
    wandb.finish()

# Base and save directories
base_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model_dynamic/tau_5.0_with_varying_step"
save_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_Laminar_Flow_Dynamic"

# Parameters
n_variations = 100
c_in_list = []
c_out_list = []

# Load all variations for training
for i in np.linspace(0, 1, n_variations):  # Match i_values used during saving
    i_str = f"{i:.2f}".replace('.', '_')  # Format i for filenames
    variation_dir = os.path.join(base_dir, f"variation_{i_str}")  # Directory for each variation
    
    t_conv = np.load(os.path.join(variation_dir, "time.npy"))
    c_out = np.load(os.path.join(variation_dir, "concentration.npy"))
    input_func = np.load(os.path.join(variation_dir, "input_function_interp.npy"))

    c_in_list.append(torch.tensor(input_func, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    c_out_list.append(torch.tensor(c_out, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

# Prepare training data
c_in_train = torch.cat(c_in_list[:-1], dim=0)  # Use all but one for training
c_out_train = torch.cat(c_out_list[:-1], dim=0)


test_time_path = os.path.join(base_dir, "time_test.npy")
test_concentration_path = os.path.join(base_dir, "concentration_test.npy")
test_input_function_path = os.path.join(base_dir, "input_test.npy")


# Print the paths for debugging
print("Test time file path:", test_time_path)
print("Test concentration file path:", test_concentration_path)
print("Test input function file path:", test_input_function_path)

# Check if the files exist and print their existence status
print("Test time file exists:", os.path.exists(test_time_path))
print("Test concentration file exists:", os.path.exists(test_concentration_path))
print("Test input function file exists:", os.path.exists(test_input_function_path))

# Raise an error if any file is missing
if not (os.path.exists(test_time_path) and os.path.exists(test_concentration_path) and os.path.exists(test_input_function_path)):
    raise FileNotFoundError("One or more test file components are missing at the specified path.")

# Load the specified test file
test_t_conv = torch.tensor(np.load(test_time_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
test_c_out = torch.tensor(np.load(test_concentration_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
test_input_func = torch.tensor(np.load(test_input_function_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
print(test_input_func.shape)
# Debugging the loaded shapes
for i, (c_in, c_out) in enumerate(zip(c_in_list, c_out_list)):
    print(f"Training variation {i}: c_in shape = {c_in.shape}, c_out shape = {c_out.shape}")

# Prepare training data and print their shapes
c_in_train = torch.cat(c_in_list[:-1], dim=0)  # Use all but one for training
c_out_train = torch.cat(c_out_list[:-1], dim=0)
print("c_in_train shape:", c_in_train.shape)
print("c_out_train shape:", c_out_train.shape)

# Test file paths
test_time_path = os.path.join(base_dir, "time_test.npy")
test_concentration_path = os.path.join(base_dir, "concentration_test.npy")
test_input_function_path = os.path.join(base_dir, "input_test.npy")

# Print paths for debugging
print("Test time file path:", test_time_path)
print("Test concentration file path:", test_concentration_path)
print("Test input function file path:", test_input_function_path)

# Check file existence
print("Test time file exists:", os.path.exists(test_time_path))
print("Test concentration file exists:", os.path.exists(test_concentration_path))
print("Test input function file exists:", os.path.exists(test_input_function_path))

# Load test files and print their shapes
if os.path.exists(test_time_path) and os.path.exists(test_concentration_path) and os.path.exists(test_input_function_path):
    test_t_conv = torch.tensor(np.load(test_time_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    test_c_out = torch.tensor(np.load(test_concentration_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    test_input_func = torch.tensor(np.load(test_input_function_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    print("test_t_conv shape:", test_t_conv.shape)
    print("test_c_out shape:", test_c_out.shape)
    print("test_input_func shape:", test_input_func.shape)
else:
    raise FileNotFoundError("One or more test file components are missing at the specified path.")

# Debugging final shapes
print("Final shapes:")
print("c_in_train shape:", c_in_train.shape)
print("c_out_train shape:", c_out_train.shape)
print("test_input_func shape:", test_input_func.shape)
print("test_c_out shape:", test_c_out.shape)

# Create datasets
train_dataset = TensorDataset(c_in_train, c_out_train)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

test_dataset = TensorDataset(test_input_func, test_c_out)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model initialization
model = RTDModule(
    kernel_sizes=[151],
    kernel_times=[(0.0, 75)],
    learning_rate=1e-4,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

# Logger setup
wandb_logger = pl_loggers.WandbLogger(project="nRTD", log_model=True)

# Training the model
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=1000,
    logger=wandb_logger,
    deterministic=True,
)

# Train and test the model
trainer.fit(model, train_loader)
results = trainer.test(model, test_loader)
t_e_1=75
c_conv = model(c_in_train)  
E = model.net.E[0]
t_E =torch.linspace(0, float(t_e_1), int(model.kernel_sizes[0]))
E = E/E.max()

# Expected E calculation
t_E_np = np.linspace(0, 75, 151)
def expected_formula(t):
    return np.where(t >= 2.5, (2.5**2) / (2 * (t**3)), 0)
E_expected = expected_formula(t_E_np)
E_expected /= E_expected.max()

# Plot E Predicted vs. Expected
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
# np.save(os.path.join(save_dir, "E_predicted.npy"), E.detach().numpy())
# np.save(os.path.join(save_dir, "t_E_predicted.npy"), t_E.numpy())
# np.save(os.path.join(save_dir, "E_expected.npy"), E_expected)
# np.save(os.path.join(save_dir, "t_E_expected.npy"), t_E_np)
