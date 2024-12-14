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
import sys
import os
module_path = os.path.expanduser("lib")
sys.path.append(module_path)

module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)

import os
import datetime
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch
import os
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots import make_square_subplots
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots.figures import Elsevier_Sizes
from ICIW_Plots import make_square_ax, cm2inch
save_dir="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Plots"

tau_5_dir_lam="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model/tau_5.0_disc_200_100s"
t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir_lam, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir_lam, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/c_conv_in.npy')
predicted_time = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/t_E_predicted.npy')
predicted_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/E_predicted.npy')
expected_E = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_021_Laminar_Flow_Model/E_expected.npy')
######
Bo_1 = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Dispersion_Model/Bo1200dis_240s_1412'
t_conv_Bo_1 = torch.tensor(np.load(os.path.join(Bo_1, 'time_240.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_Bo_1 = torch.tensor(np.load(os.path.join(Bo_1, 'concentration_240.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_1_1412/c_conv_in_Bo_1.npy')
predicted_time_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_1_1412/t_E_predicted_Bo_1.npy')
predicted_E_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_1_1412/E_predicted_Bo_1.npy')
expected_E_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_1_1412/E_expected_Bo_1.npy')
########
Bo_ = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Dispersion_Model/Bo1200dis_240s_1412'
t_conv_Bo_1 = torch.tensor(np.load(os.path.join(Bo_1, 'time_240.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_Bo_1 = torch.tensor(np.load(os.path.join(Bo_1, 'concentration_240.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_023_Adler_havarka_Model/c_conv_in_Bo_1.npy')
predicted_time_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_1_1412/t_E_predicted_Bo_1.npy')
predicted_E_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_1_1412/E_predicted_Bo_1.npy')
expected_E_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_1_1412/E_expected_Bo_1.npy')


t_conv_tau_1d = t_conv_tau.squeeze().numpy()  
c_out_tau_1d = c_out_tau.squeeze().numpy()   
predicted_c_1d = predicted_c.squeeze()     
  
t_conv_Bo_1 = t_conv_Bo_1.squeeze().numpy()  
c_out_Bo_1 = c_out_Bo_1.squeeze().numpy()   
predicted_c_Bo_1 = predicted_c_Bo_1.squeeze()       


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
    ])
axs[0, 0].plot(
    t_conv_tau_1d,  # Ensure this is 1D
    predicted_c_1d,
    label=r"Predicted",
    color="black",
)
axs[0, 0].plot(
    t_conv_tau_1d,
    c_out_tau_1d,
    label=r"Expected",
    color="purple",
    linestyle="--",
)

axs[0, 1].plot(
    predicted_time,  # Ensure this is also 1D
    predicted_E,
    label=r"noisy_100",
    color=ICIWcolors.KELLYGREEN,linestyle="-.",
)

axs[0, 1].plot(
    predicted_time,  # Ensure this is also 1D
    expected_E,
    label=r"noisy_100",
    color="orange",
)
####
axs[1, 0].plot(
    t_conv_Bo_1,  # Ensure this is 1D
    predicted_c_Bo_1,
    label=r"Predicted",
    color="black",
)
axs[1, 0].plot(
    t_conv_Bo_1,
    c_out_Bo_1,
    label=r"Expected",
    color="purple",
    linestyle="--",
)

axs[1, 1].plot(
    predicted_time_Bo_1,  # Ensure this is also 1D
    predicted_E_Bo_1,
    label=r"noisy_100",
    color=ICIWcolors.KELLYGREEN,linestyle="-.",
)

axs[1, 1].plot(
    predicted_time_Bo_1,  # Ensure this is also 1D
    expected_E_Bo_1,
    label=r"noisy_100",
    color="orange",
)
plt.show()