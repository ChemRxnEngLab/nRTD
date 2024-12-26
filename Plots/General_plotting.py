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
Bo_1 = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Dispersion_Model/Bo_100_50'
t_conv_Bo_1 = torch.tensor(np.load(os.path.join(Bo_1, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_Bo_1 = torch.tensor(np.load(os.path.join(Bo_1, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_100/c_conv_in_Bo_100.npy')
predicted_time_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_100/t_E_predicted_Bo_10.npy')
predicted_E_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_100/E_predicted_Bo_10.npy')
expected_E_Bo_1 = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_022_Dispersion_Model/Bo_100/E_expected_Bo_10.npy')
########
Adler = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Adler_havarka_Model/tau_a_val_1_tau_p_val_2_tau_m_val_0.4000000000000001_beta_val_0.1_26_11'
t_conv_Adler = torch.tensor(np.load(os.path.join(Adler, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_Adler = torch.tensor(np.load(os.path.join(Adler, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c_Adler = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_023_Adler_havarka_Model/c_conv_in.npy')
predicted_time_Adler = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_023_Adler_havarka_Model/t_E_predicted_tau_a_val_1.npy')
predicted_E_Adler = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_023_Adler_havarka_Model/E_predicted_tau_a_val_1.npy')
expected_E_Adler= np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_023_Adler_havarka_Model/E_expected_tau_a_val_1.npy')
###########
Cholete = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Cholete_Model/beta0.1_n'
t_conv_Cholete = torch.tensor(np.load(os.path.join(Cholete, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_Cholete = torch.tensor(np.load(os.path.join(Cholete, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c_Cholete = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_024_Cholete_Model/c_conv_in.npy')
predicted_time_Cholete = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_024_Cholete_Model/t_E_predicted_01.npy')
predicted_E_Cholete= np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_024_Cholete_Model/E_predicted_01.npy')
expected_E_Cholete= np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_024_Cholete_Model/E_expected_01.npy')
###########
Unified = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Unified_time_delay'
t_conv_Unified = torch.tensor(np.load(os.path.join(Unified, 'time_J5_tau3_alpha0.2_200.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_Unified = torch.tensor(np.load(os.path.join(Unified, 'concentration_J5_tau3_alpha0.2_200.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
predicted_c_Unified = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/c_conv_in_j5_3.npy')
predicted_time_Unified = np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/t_E_predicted_J5_tau_3.npy')
predicted_E_Unified= np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/E_predicted_J5_tau_3.npy')
expected_E_Unified= np.load('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Convolution_script/Convolution_025_Unified_time_Model/E_expected_J5_tau_3.npy')
n_disc=100
t_input = torch.linspace(0, 50, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 5] = 1
c_in[1::2, :, t_input < 5] = 1

n_disc_Bo=100
t_input_Bo = torch.linspace(0, 70, n_disc)
c_in_Bo = torch.zeros((1, 1, n_disc))
c_in_Bo[::2, :, t_input_Bo > 5] = 1
c_in_Bo[1::2, :, t_input_Bo < 5] = 1

n_disc_Adler=100
t_input_Adler = torch.linspace(0, 35, n_disc)
c_in_Adler = torch.zeros((1, 1, n_disc))
c_in_Adler[::2, :, t_input_Adler > 5] = 1
c_in_Adler[1::2, :, t_input_Adler < 5] = 1

n_disc_Cholete=100
t_input_Cholete = torch.linspace(0,25, n_disc)
c_in_Cholete = torch.zeros((1, 1, n_disc))
c_in_Cholete[::2, :, t_input_Cholete > 5] = 1
c_in_Cholete[1::2, :, t_input_Cholete < 5] = 1

n_disc_Unified=100
t_input_Unified = torch.linspace(0, 35, n_disc)
c_in_Unified = torch.zeros((1, 1, n_disc))
c_in_Unified[::2, :, t_input_Unified > 5] = 1
c_in_Unified[1::2, :, t_input_Unified < 5] = 1

t_conv_tau_1d = t_conv_tau.squeeze().numpy()  
c_out_tau_1d = c_out_tau.squeeze().numpy()   
predicted_c_1d = predicted_c.squeeze()     
  
t_conv_Bo_1 = t_conv_Bo_1.squeeze().numpy()  
c_out_Bo_1 = c_out_Bo_1.squeeze().numpy()   
predicted_c_Bo_1 = predicted_c_Bo_1.squeeze()       

t_conv_Adler = t_conv_Adler.squeeze().numpy()  
c_out_Adler = c_out_Adler.squeeze().numpy()   
predicted_c_Adler = predicted_c_Adler.squeeze()   

t_conv_Cholete = t_conv_Cholete.squeeze().numpy()  
c_out_Cholete = c_out_Cholete.squeeze().numpy()   
predicted_c_Cholete = predicted_c_Cholete.squeeze()   

t_conv_Unified = t_conv_Unified.squeeze().numpy()  
c_out_Unified = c_out_Unified.squeeze().numpy()   
predicted_c_Unified = predicted_c_Unified.squeeze()   

plt.style.use("ICIWstyle")

fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 25 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(3, 2),  
    h_sep=1.3 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=True,
    sharey=False,
    xlabel=[r"$t$ / $s$", r"$t$ / $s$"], 
    ylabel=[
        [r"$C$ / $1$", r"$E$ / $1$"],   [r"$C$ / $1$", r"$E$ / $1$"],[r"$C$ / $1$", r"$E$ / $1$"]
    ])

axs[0, 0].plot(
    t_conv_tau_1d,
    c_out_tau_1d,
    label=r"$C_{Laminar}(t)$",
   color=ICIWcolors.DRAB
)
axs[0, 0].plot(
    t_conv_tau_1d,  # Ensure this is 1D
    predicted_c_1d,
    label=r"$\hat{C}_{Laminar}(t)$",
    color="purple",
    linestyle="--"
)

axs[0, 0].plot(t_input, c_in[0, 0, :].numpy(), label=r"$C_0(t)$", color=ICIWcolors.CERULEAN)
# axs[0, 0].text(-0.1, 0.87, "(a)", transform=axs[0, 0].transAxes, fontsize=11)  # Add label

axs[0, 1].plot(
    predicted_time,  # Ensure this is also 1D
    expected_E, label=r"$E_{Laminar}(t)$", color=ICIWcolors.KELLYGREEN
)
axs[0, 1].plot(
    predicted_time,  # Ensure this is also 1D
    predicted_E,label=r"$\hat{E}_{Laminar}(t)$", color="black", linestyle="--"
)
axs[0, 0].set_xlim((0, 30)) 
axs[0, 1].set_xlim((0, 20)) 
####
axs[1, 0].plot(
    t_conv_Bo_1,
    c_out_Bo_1,
    label=r"$C_{Dispersion}(t)$",
   color=ICIWcolors.DRAB
)
axs[1, 0].plot(
    t_conv_Bo_1,  # Ensure this is 1D
    predicted_c_Bo_1,
    label=r"$\hat{C}_{Dispersion}(t)$",
    color="purple",
    linestyle="--")


axs[1, 0].plot(t_input_Bo, c_in_Bo[0, 0, :].numpy(), label=r"$C_0(t)$", color=ICIWcolors.CERULEAN)



axs[1, 1].plot(
    predicted_time_Bo_1,  # Ensure this is also 1D
    expected_E_Bo_1,
label=r"$E_{Dispersion}(t)$", color=ICIWcolors.KELLYGREEN
)

axs[1, 1].plot(
    predicted_time_Bo_1,  # Ensure this is also 1D
    predicted_E_Bo_1,
label=r"$\hat{E}_{Dispersion}(t)$", color="black", linestyle="--"
)

axs[1, 0].set_xlim((0, 80)) 
axs[1, 1].set_xlim((0, 80)) 

##########
axs[2, 0].plot(
    t_conv_Adler,
    c_out_Adler,
    label=r"$C_{Adler}(t)$",
   color=ICIWcolors.DRAB
)
axs[2, 0].plot(
    t_conv_Adler,  # Ensure this is 1D
    predicted_c_Adler,
    label=r"$\hat{C}_{Adler}(t)$",
    color="purple",
    linestyle="--"
)

axs[2, 0].plot(t_input_Adler, c_in_Adler[0, 0, :].numpy(), label=r"$C_0(t)$", color=ICIWcolors.CERULEAN)

axs[2, 1].plot(
    predicted_time_Adler,  # Ensure this is also 1D
    expected_E_Adler,
label=r"$E_{Adler}(t)$", color=ICIWcolors.KELLYGREEN
)
axs[2, 1].plot(
    predicted_time_Adler,  # Ensure this is also 1D
    predicted_E_Adler,
label=r"$\hat{E}_{Adler}(t)$", color="black", linestyle="--"
)
axs[0, 0].legend(loc="best")
axs[0, 1].legend(loc="best")
axs[1,0].legend(loc="best")
axs[1, 1].legend(loc="best")
axs[2, 0].legend(loc="best")
axs[2, 1].legend(loc="best")
axs[2, 0].set_xlim((0, 30)) 
axs[2, 1].set_xlim((0, 30)) 
plt.savefig(os.path.join(save_dir, f"lit_models.png"), dpi=300)
plt.show()

plt.style.use("ICIWstyle")

#######################################

fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 25 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(2, 2),  
    h_sep=1.3 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=True,
    sharey=False,
    xlabel=[r"$t$ / $s$", r"$t$ / $s$"], 
    ylabel=[
        [r"$C$ / $1$", r"$E$ / $1$"],   [r"$C$ / $1$", r"$E$ / $1$"],
    ])

axs[0, 0].plot(
    t_conv_Cholete,
    c_out_Cholete,
    label=r"$C_{Cholete}(t)$",
   color=ICIWcolors.DRAB
)
axs[0, 0].plot(
    t_conv_Cholete,  # Ensure this is 1D
    predicted_c_Cholete,
    label=r"$\hat{C}_{Cholete}(t)$",
    color="purple",
    linestyle="--"
)

axs[0, 0].plot(t_input, c_in[0, 0, :].numpy(), label=r"$C_0(t)$", color=ICIWcolors.CERULEAN)
axs[0, 1].plot(
    predicted_time_Cholete,  # Ensure this is also 1D
    expected_E_Cholete, label=r"$E_{Cholete}(t)$", color=ICIWcolors.KELLYGREEN
)
axs[0, 1].plot(
    predicted_time_Cholete,  # Ensure this is also 1D
    predicted_E_Cholete,label=r"$\hat{E}_{Cholete}(t)$", color="black", linestyle="--"
)
####
axs[1, 0].plot(
    t_conv_Unified,
    c_out_Unified,
    label=r"$C_{Unified}(t)$",
    color=ICIWcolors.DRAB
)
axs[1, 0].plot(
    t_conv_Unified,  # Ensure this is 1D
    predicted_c_Unified,
    label=r"$\hat{C}_{Unified}(t)$",
    color="purple",
    linestyle="--")


axs[1, 0].plot(t_input_Unified, c_in_Unified[0, 0, :].numpy(), label=r"$x_0(t)$", color=ICIWcolors.CERULEAN)



axs[1, 1].plot(
    predicted_time_Unified,  # Ensure this is also 1D
    expected_E_Unified,
label=r"$E_{Unified}(t)$", color=ICIWcolors.KELLYGREEN
)

axs[1, 1].plot(
    predicted_time_Unified,  # Ensure this is also 1D
    predicted_E_Unified,
label=r"$\hat{E}_{Unified}(t)$", color="black", linestyle="--"
)

axs[1, 0].set_xlim((0, 15)) 
axs[1, 1].set_xlim((0, 15)) 
axs[0, 0].legend(loc="best")
axs[0, 1].legend(loc="best")
axs[1,0].legend(loc="best")
axs[1, 1].legend(loc="best")
plt.savefig(os.path.join(save_dir, f"lit_models_2.png"), dpi=300)
plt.show()


