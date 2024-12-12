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

import os
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch
import torch
import numpy as np
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots import make_rect_ax

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