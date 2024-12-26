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

fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 10 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(1, 2),  
    h_sep=1.39 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=False,
    sharey=False,
    xlabel=["$Epoch/1$","$t/1$"], 
    ylabel=
        [["$Test/loss$","$C/1$"]]
    
)


axs[0, 0].plot(files, error,'o-', color=ICIWcolors.FLAME)
axs[0, 1].plot(
    t_conv_tau_1d,
    c_out_tau_1d,
    label=r"$x(t)$",
    color="purple",
    )
axs[0, 1].plot(
    t_conv_tau_1d,  # Ensure this is 1D
    predicted_c_1d,
    label=r"$\hat{x}(t)$",
    color=ICIWcolors.KELLYGREEN,linestyle="--",
)

axs[0, 1].plot(
    t_conv_tau_1d,  # Ensure this is also 1D
    c_conv_1_1d,
    label=r"$\hat{x}_{noisy,1}(t)$",
    color=ICIWcolors.FLAME,linestyle="--",
)

axs[0, 1].plot(
    t_conv_tau_1d,  # Ensure this is also 1D
    c_conv_100_first_config,
    label=r"$\hat{x}_{noisy,100}(t)$",
    color="#A0CFCF",linestyle="--",
)

axs[0, 1].set_xlim((0, 55))
# axs[1, 0].set_ylim((-0.1, 1.1))
axs[0, 0].set_yscale('log')

#ax.plot()
axs[0, 1].legend(loc="best")  # For the second subplot
plt.savefig(os.path.join(save_dir, f"noisy.png"), dpi=300)
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

epoch=[100,1000,5000,10000,15000,20000]
error_epoch=[0.0043839,0.00030317,0.0000012918,7.96e-8,7.4542e-8,7.4542e-8]

file_path = "/Users/tuanaoyuncu/Desktop/epoch_training.csv"
data = pd.read_csv(file_path)
selected_values = data.iloc[:, 0].tolist()  
train_loss_values_63 = data.iloc[:, 4].tolist() 

plt.style.use("ICIWstyle")

fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 10 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(1, 2),  
    h_sep=1.39 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=False,
    sharey=False,
    xlabel=["$Epoch/1$","$Epoch/1$"], 
    ylabel=
        [["$Test/loss$","$Train/loss$"]]
    
)

axs[0, 0].plot(
    epoch,
    error_epoch,'o-',
    color=ICIWcolors.FLAME,
)

axs[0, 0].set_yscale('log')
axs[0, 1].plot(selected_values, train_loss_values_63,color=ICIWcolors.KELLYGREEN)
axs[0, 1].set_yscale('log')
plt.savefig(os.path.join(save_dir, f"epoch_1512.png"), dpi=300)
plt.show()
#########

LR_2=[10,1,1e-1,1e-2,1e-3,1e-4,1e-5]
error_LR_2=[8.2e-9,8.4e-9,8.5e-9,8.1e-9,8.7e-9,8.3e-9,9.5e-9]
epoch_2=[10000,50000,100000,150000,200000,230000,250000]
error_epoch_2=[0.0000092,8.1e-7,8.3e-8,1.9e-8,9.2e-9,8.6e-9,8.3e-9]
plt.style.use("ICIWstyle")
fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 10 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(1, 2),  
    h_sep=1.39 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=False,
    sharey=False,
    xlabel=["$Epoch/1$","$Epoch/1$"], 
    ylabel=
        [["$Test/loss$","$Train/loss$"]]
    
)

axs[0, 1].plot(
    epoch_2,
    error_epoch_2,'o-',
    color=ICIWcolors.FLAME,
)

axs[0, 1].set_yscale('log')
axs[0, 0].plot(LR_2, error_LR_2,'o-',color=ICIWcolors.KELLYGREEN)
#axs[0, 1].set_yscale('log')
axs[0, 0].set_xscale('log')
#axs[0, 0].set_yscale('log')
axs[0, 0].set_xticks(LR)
axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
axs[0, 0].set_xlim(min(LR) * 0.5, max(LR) * 2)


plt.savefig(os.path.join(save_dir, f"2nd_hypo.png"), dpi=300)
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
    sharex=False,
    sharey=False,
    xlabel=["$Epoch$","$Epoch$"], 
    ylabel=
        [["$test/loss$","$train/loss$"]]
    
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
t10 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1_001/TOA_MGA_20231020_009_000001_t_processed.npy")
x10 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1_001/TOA_MGA_20231020_009_000001_x_processed.npy")
x10 /= 224
x10_p=x10/x10.max()
t11 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_t_processed.npy")
x11 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_x_processed.npy")
x11 /= 224
x11_p=x11/x11.max()
t4 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_t_processed.npy")
t5 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_135_C2/S_013_C2_001/TOA_MGA_20231020_013_000001_t_processed.npy")

plt.style.use("ICIWstyle")

fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 25 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=8 * cm2inch,#dimension of the plots
    ax_layout=(1, 1),  
    h_sep=1.3 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=True,
    sharey=False,
    xlabel= "$t/s$",
    ylabel=[
        "$C/1$"
    ]
)
#     sharex=True,
#     sharey=True,
#     xlabel=["$x_1$", "$x_2$"], 
#     ylabel=[["$y_1$"],["$y_1$"],["$y_1$"]]
         
    
# )

axs[0, 0].plot(
    t4, x10_p,
    label=r"$(1)$",
    color=ICIWcolors.KELLYGREEN,
)
axs[0, 0].plot(
    t4, x11_p,
    label=r"$(2)$",
    color=ICIWcolors.FLAME,
)
# axs[0, 0].set_xscale('$Epoch$')
# axs[0, 0].set_yscale('$test/loss$')
# axs[0, 0].set_xticks(LR)
# axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
axs[0, 0].set_xlim(0,20)
plt.savefig(os.path.join(save_dir, f"try_exp.png"), dpi=300)
plt.show()

####3
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
axs[0, 0].set_xscale('$Epoch$')
axs[0, 0].set_yscale('$test/loss$')
axs[0, 0].set_xticks(LR)
axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
axs[0, 0].set_xlim(min(LR) * 0.5, max(LR) * 2)
plt.savefig(os.path.join(save_dir, f"test.png"), dpi=300)
plt.show()


#######################
###########################

