import matplotlib.pyplot as plt

import sys
import os
module_path = os.path.expanduser("lib")
sys.path.append(module_path)
import torch

import numpy as np

# module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
# sys.path.append(module_path)

module_path = r"D:\Tuana\nRTD\lib"
sys.path.append(module_path)
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
from nRTD.rtd_fitting_2 import RTDModule
from nRTD.rtd_net_4 import RTDNet
from lightning.pytorch import loggers as pl_loggers
import os
import datetime
import sympy as sp
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch
import os
import datetime
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch
import torch
import numpy as np
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots import make_rect_ax
from ICIW_Plots import make_square_subplots
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import numpy as np
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots.figures import Elsevier_Sizes
import datetime
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch
save_dir="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Plots"

tau_5_dir_lam="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model/tau_5.0_disc_200_100s"
t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir_lam, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir_lam, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/c_conv_in.npy')
predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/t_E_predicted.npy')
t_conv_100=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method/Results_Noisy_Data/t_E_100.npy")
c_conv_100=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method/Results_Noisy_Data/c_conv_100.npy")
t_conv_1=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method/Results_Noisy_Data/t_E_1.npy")
c_conv_1=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Noisy_data/Sqrt_Method/Results_Noisy_Data/c_conv_1.npy")
t_conv_tau_1d = t_conv_tau.squeeze().numpy()  # Shape becomes (200,)
c_out_tau_1d = c_out_tau.squeeze().numpy()   # Shape becomes (200,)
predicted_c_1d = predicted_c.squeeze()       # Ensure compatibility for numpy array
c_conv_100_1d = c_conv_100.squeeze()         # Ensure compatibility for numpy array
c_conv_100_first_config = c_conv_100_1d[0, :]
c_conv_1_1d = c_conv_1.squeeze() 


plt.style.use("ICIWstyle")
fig = plt.figure()
ax = make_rect_ax(
    fig,
    ax_width=7.3 * cm2inch,
    ax_height=5 * cm2inch,
    xlabel="$t$ / $s$",
    ylabel="$x$ / $1$",
)

ax.plot(
    t_conv_tau_1d,  # Ensure this is 1D
    predicted_c_1d,
    label=r"Conv_model",
    color="black",
)

ax.plot(
    t_conv_tau_1d,
    c_out_tau_1d,
    label=r"$Litrature$",
    color="purple",
    linestyle="--",
)


ax.plot(
    t_conv_tau_1d,  # Ensure this is also 1D
    c_conv_100_first_config,
    label=r"noisy_100",
    color=ICIWcolors.KELLYGREEN,linestyle="-.",
)

ax.plot(
    t_conv_tau_1d,  # Ensure this is also 1D
    c_conv_1_1d,
    label=r"noisy_100",
    color="orange",
)

ax.set_xlim((0, 50))
ax.set_ylim((-0.1, 1.1))
ax.legend(loc="best")

plt.show()
print(t_conv_tau_1d.shape)
print(predicted_c_1d.shape)
print(c_conv_100_1d.shape)

files=[1,20,40,60,80,100]
error=[4.8193e-5,2.0946e-6,1.0367e-6,7.7302e-7,6.4032e-7,5.2123e-7]#7.5e-8

plt.style.use("ICIWstyle")
fig = plt.figure()
ax = make_rect_ax(
    fig,
    ax_width=7.3 * cm2inch,
    ax_height=5 * cm2inch,
    xlabel="$number of dataset$",
    ylabel="$test/loss / 1$",
)

# ax.plot.semilogy(
#     files, 
#     error,
#     label=r"Conv_model",
#     color="black",
# )

ax.semilogy(files, error,'o-', label="Conv_model", color="black")
ax.plot()

ax.legend(loc="best")
plt.show()

LR=[10,1,1e-1,1e-2,1e-3,1e-4,1e-5]
error_2=[7.4647e-8,7.46e-8,7.4541e-8,7.4523e-8,7.4537e-8,1.1904e-7,0.00067261]#7.5e-8

# plt.style.use("ICIWstyle")

# fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 24 * cm2inch))  # Increased figure height for better spacing
# axs = make_square_subplots(
#     fig=fig,
#     ax_width=7.5 * cm2inch,
#     ax_layout=(3, 2),  
#     h_sep=1.1 * cm2inch,  
#     v_sep=0.1 * cm2inch, 
#     sharex=True,
#     sharey=False,
#     xlabel=["$x_1$", "$x_2$"], 
#     ylabel=[
#         ["$y_1$", "$y_2$"],  
#         ["$y_3$", "$y_4$"], 
#         ["$y_5$", "$y_6$"],  
#     ]
# )
# #     sharex=True,
# #     sharey=True,
# #     xlabel=["$x_1$", "$x_2$"], 
# #     ylabel=[["$y_1$"],["$y_1$"],["$y_1$"]]
         
    
# # )

# axs[0, 0].plot(
#     LR,
#     error_2,
#     label=r"$\hat{x}(t)$",
#     color="purple",
#     linestyle="--"
# )
# axs[0, 0].set_xscale('log')
# axs[0, 0].set_yscale('log')
# axs[0, 0].set_xticks(LR)
# axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
# axs[0, 0].set_xlim(min(LR) * 0.5, max(LR) * 2)
# axs[0, 0].plot()
# plt.savefig(os.path.join(save_dir, f"test.png"), dpi=300)
# plt.show()
#######
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "/Users/tuanaoyuncu/Desktop/trainandloss.csv"  # Replace with the actual path to your CSV file
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the actual path to your CSV file

# Load the CSV, specifying no header and using a custom column name list
data = pd.read_csv(file_path, header=None, delimiter=',', 
                   names=["epoch", "playful-sun-1763 - _step", "playful-sun-1763 - _step__MIN", "playful-sun-1763 - _step__MAX", 
                          "playful-sun-1763 - train/loss", "playful-sun-1763 - train/loss__MIN", "playful-sun-1763 - train/loss__MAX",
                          "true-pond-1762 - _step", "true-pond-1762 - _step__MIN", "true-pond-1762 - _step__MAX", 
                          "true-pond-1762 - train/loss", "true-pond-1762 - train/loss__MIN", "true-pond-1762 - train/loss__MAX",
                          "twilight-mountain-1759 - _step", "twilight-mountain-1759 - _step__MIN", "twilight-mountain-1759 - _step__MAX",
                          "twilight-mountain-1759 - train/loss", "twilight-mountain-1759 - train/loss__MIN", "twilight-mountain-1759 - train/loss__MAX",
                          "zany-aardvark-1758 - _step", "zany-aardvark-1758 - _step__MIN", "zany-aardvark-1758 - _step__MAX",
                          "zany-aardvark-1758 - train/loss", "zany-aardvark-1758 - train/loss__MIN", "zany-aardvark-1758 - train/loss__MAX",
                          "mild-universe-1757 - _step", "mild-universe-1757 - _step__MIN", "mild-universe-1757 - _step__MAX",
                          "mild-universe-1757 - train/loss", "mild-universe-1757 - train/loss__MIN", "mild-universe-1757 - train/loss__MAX",
                          "brisk-haze-1756 - _step", "brisk-haze-1756 - _step__MIN", "brisk-haze-1756 - _step__MAX",
                          "brisk-haze-1756 - train/loss", "brisk-haze-1756 - train/loss__MIN", "brisk-haze-1756 - train/loss__MAX",
                          "winter-dust-1753 - _step", "winter-dust-1753 - _step__MIN", "winter-dust-1753 - _step__MAX",
                          "winter-dust-1753 - train/loss", "winter-dust-1753 - train/loss__MIN", "winter-dust-1753 - train/loss__MAX"])

# Extract the relevant columns for epochs and train loss
epochs = data["epoch"]
print(epochs)
# List of columns for each model's train/loss
train_loss_columns = [
    "playful-sun-1763 - train/loss",
    "true-pond-1762 - train/loss",
    "twilight-mountain-1759 - train/loss",
    "zany-aardvark-1758 - train/loss",
    "mild-universe-1757 - train/loss",
    "brisk-haze-1756 - train/loss",
    "winter-dust-1753 - train/loss"
]

# Plot each train/loss against epochs
plt.figure(figsize=(12, 8))

for column in train_loss_columns:
    plt.plot(epochs, data[column], label=column.split(" - ")[0])

# Customize plot
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Train Loss")
plt.title("Epochs vs Train Loss for Various Models")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()


#######
plt.style.use("ICIWstyle")

fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 10 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7.5 * cm2inch,#dimension of the plots
    ax_layout=(1, 2),  
    h_sep=1.3 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=True,
    sharey=False,
    xlabel=["$x_1$", "$x_2$"], 
    ylabel=[
        ["$y_1$", "$y_2$"],   
    ]
)

axs[0, 0].plot(
    LR,
    error_2,'o-',
    label=r"$\hat{x}(t)$",
    color=ICIWcolors.FLAME,
)
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')
axs[0, 0].set_xticks(LR)
axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
axs[0, 0].set_xlim(min(LR) * 0.5, max(LR) * 2)


plt.savefig(os.path.join(save_dir, f"test.png"), dpi=300)
plt.show()
#######################

plt.style.use("ICIWstyle")

fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 25 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7.5 * cm2inch,#dimension of the plots
    ax_layout=(3, 2),  
    h_sep=1.3 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=True,
    sharey=False,
    xlabel=["$x_1$", "$x_2$"], 
    ylabel=[
        ["$y_1$", "$y_2$"],   ["$y_1$", "$y_2$"],["$y_1$", "$y_2$"]
    ]
)
#     sharex=True,
#     sharey=True,
#     xlabel=["$x_1$", "$x_2$"], 
#     ylabel=[["$y_1$"],["$y_1$"],["$y_1$"]]
         
    
# )

axs[0, 0].plot(
    LR,
    error_2,'o-',
    label=r"$\hat{x}(t)$",
    color="purple",
)
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')
axs[0, 0].set_xticks(LR)
axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
axs[0, 0].set_xlim(min(LR) * 0.5, max(LR) * 2)



plt.savefig(os.path.join(save_dir, f"test.png"), dpi=300)
plt.show()


#######################
fig = plt.figure()
ax = make_rect_ax(
    fig,
    ax_width=7.3 * cm2inch,
    ax_height=5 * cm2inch,
    xlabel="$Learning rate$",
    ylabel="$test/loss / 1$",
)

ax.plot(LR, error_2,'o-', label="Conv_model", color="black")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(LR)
ax.set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])


ax.set_xlim(min(LR) * 0.5, max(LR) * 2)
ax.plot()

ax.legend(loc="best")
plt.show()
###########################
print()
from ICIW_Plots import make_square_subplots
import matplotlib.pyplot as plt

fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 23 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=6 * cm2inch,
    ax_layout=(3, 2),  
    h_sep=2. * cm2inch,  
    v_sep=1.3 * cm2inch, 
    sharex=True,
    sharey=False,
    xlabel=["$x_1$", "$x_2$"], 
    ylabel=[
        ["$y_1$", "$y_2$"],  
        ["$y_3$", "$y_4$"], 
        ["$y_5$", "$y_6$"],  
    ]
)

for i in range(c_out.size(1)):
    axs[0, 0].plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label=r"$x(t)$",
        color="green"
    )
axs[0, 0].plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label=r"$\hat{x}(t)$",
    color="purple",
    linestyle="--"
)
axs[0, 0].set_ylabel(r"$x$ / $1$")
axs[0, 0].legend(loc="best")
axs[0, 0].set_xlim((0, 25))
axs[0, 0].set_ylim((-0.1, 1.1))
axs[0, 1].plot(
    expected_time, expected_E, label=r"$E(t)$", color="green"
)
axs[0, 1].plot(
    predicted_time, predicted_E, label=r"$\hat{E}(t)$", color="black", linestyle="--"
)
axs[0, 1].set_xlabel(r"$t$ / $s$")
axs[0, 1].set_ylabel(r"$E$ / $1$")
axs[0, 1].legend(loc="best")
axs[0, 1].set_xlim((0, 25))
axs[0, 1].set_ylim((-0.1, 1.1))
current_date = datetime.datetime.now().strftime("%Y%m%d")
plt.savefig(os.path.join(save_dir, f"test_3_{current_date}.png"), dpi=300)
plt.show()
