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
Unified = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Unified_time_delay'
t_conv_Unified = torch.tensor(np.load(os.path.join(Unified, 'time_J2_tau3_alpha0.2_200.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_Unified = torch.tensor(np.load(os.path.join(Unified, 'concentration_J2_tau3_alpha0.2_200.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c_Unified = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/c_conv_in_j2_3.npy')
predicted_time_Unified = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/t_E_predicted_J2_tau_3.npy')
predicted_E_Unified= np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/E_predicted_J2_tau_3.npy')
expected_E_Unified= np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/E_expected_J2_tau_3.npy')

Unified_2 = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Unified_time_delay'
t_conv_Unified_2 = torch.tensor(np.load(os.path.join(Unified, 'time_J5_tau3_alpha0.2_200.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_Unified_2 = torch.tensor(np.load(os.path.join(Unified, 'concentration_J5_tau3_alpha0.2_200.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c_Unified_2 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/c_conv_in_j5_3.npy')
predicted_time_Unified_2 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/t_E_predicted_J5_tau_3.npy')
predicted_E_Unified_2= np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/E_predicted_J5_tau_3.npy')
expected_E_Unified_2= np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/E_expected_J5_tau_3.npy')
save_dir="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Plots"

t_conv_Unified = t_conv_Unified.squeeze().numpy()  
c_out_Unified = c_out_Unified.squeeze().numpy()   
predicted_c_Unified = predicted_c_Unified.squeeze()  

t_conv_Unified_2 = t_conv_Unified_2.squeeze().numpy()  
c_out_Unified_2 = c_out_Unified_2.squeeze().numpy()   
predicted_c_Unified_2 = predicted_c_Unified_2.squeeze()  

n_disc_Unified=100
t_input_Unified = torch.linspace(0, 35, n_disc_Unified)
c_in_Unified = torch.zeros((1, 1, n_disc_Unified))
c_in_Unified[::2, :, t_input_Unified > 5] = 1
c_in_Unified[1::2, :, t_input_Unified < 5] = 1


fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 25 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(2, 2),  
    h_sep=1.39 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=True,
    sharey=True,
    xlabel=["$t/1$", "$t/1$"], 
    ylabel=
        ["$C/1$", "$C/1$"],  # Correctly defined 2D ylabel array
       
    
)


axs[1, 0].plot(
    t_conv_Unified_2,
    c_out_Unified_2,
    label=r"$C_{Unified_2}(t)$",
    color=ICIWcolors.DRAB
)
axs[1, 0].plot(
    t_conv_Unified_2,  # Ensure this is 1D
    predicted_c_Unified_2,
    label=r"$\hat{C}_{Unified_2}(t)$",
    color="purple",
    linestyle="--")


axs[1, 0].plot(t_input_Unified, c_in_Unified[0, 0, :].numpy(), label=r"$x_0(t)$", color=ICIWcolors.CERULEAN)



axs[1, 1].plot(
    predicted_time_Unified_2,  # Ensure this is also 1D
    expected_E_Unified_2,
label=r"$E_{Unified}(t)$", color=ICIWcolors.KELLYGREEN
)

axs[1, 1].plot(
    predicted_time_Unified_2,  # Ensure this is also 1D
    predicted_E_Unified_2,
label=r"$\hat{E}_{Unified}(t), N=5, \tau=3$", color="black", linestyle="--"
)
####
axs[0, 0].plot(
    t_conv_Unified,
    c_out_Unified,
    label=r"$C_{Unified}(t)$",
    color=ICIWcolors.DRAB
)
axs[0, 0].plot(
    t_conv_Unified,  # Ensure this is 1D
    predicted_c_Unified,
    label=r"$\hat{C}_{Unified}(t)$",
    color="purple",
    linestyle="--")


axs[0, 0].plot(t_input_Unified, c_in_Unified[0, 0, :].numpy(), label=r"$x_0(t)$", color=ICIWcolors.CERULEAN)



axs[0, 1].plot(
    predicted_time_Unified,  # Ensure this is also 1D
    expected_E_Unified,
label=r"$E_{Unified}(t)$", color=ICIWcolors.KELLYGREEN
)

axs[0, 1].plot(
    predicted_time_Unified,  # Ensure this is also 1D
    predicted_E_Unified,
label=r"$\hat{E}_{Unified}(t),N=2, \tau=3$", color="black", linestyle="--"
)


axs[0, 0].set_xlim((0, 20)) 
axs[0, 1].set_xlim((0, 15)) 
axs[1, 0].set_xlim((0, 20)) 
axs[1, 1].set_xlim((0, 15)) 
axs[0, 0].legend(loc="best")
axs[0, 1].legend(loc="best")
axs[1,0].legend(loc="best")
axs[1, 1].legend(loc="best")
plt.savefig(os.path.join(save_dir, f"unified.png"), dpi=300)
plt.show()
