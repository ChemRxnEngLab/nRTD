import matplotlib.pyplot as plt
import sys
import os
module_path = os.path.expanduser("lib")
sys.path.append(module_path)
import torch
import numpy as np

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
import numpy.typing as npt




base_dir = "/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Disc_variation"
t_expected_path_100 = f"{base_dir}/CNN_1st/Disc_100/t_E_expected_first_layer_100.npy"
t_predicted_path_100 = f"{base_dir}/CNN_1st/Disc_100/t_E_predicted_first_layer_100.npy"
E_expected_path_100 = f"{base_dir}/CNN_1st/Disc_100/E_expected_first_layer_100.npy"
E_predicted_path_100 = f"{base_dir}/CNN_1st/Disc_100/E_predicted_first_layer_100.npy"

t_expected_path_200 = f"{base_dir}/CNN_1st/Disc_200/t_E_expected_first_layer_200.npy"
t_predicted_path_200 = f"{base_dir}/CNN_1st/Disc_200/t_E_predicted_first_layer_200.npy"
E_expected_path_200 = f"{base_dir}/CNN_1st/Disc_200/E_expected_first_layer_200.npy"
E_predicted_path_200 = f"{base_dir}/CNN_1st/Disc_200/E_predicted_first_layer_200.npy"

t_expected_path_100 = np.load(t_expected_path_100)
t_predicted_path_100 = np.load(t_predicted_path_100)
E_expected_path_100 = np.load(E_expected_path_100)
E_predicted_path_100 = np.load(E_predicted_path_100)

t_expected_path_200 = np.load(t_expected_path_200)
t_predicted_path_200 = np.load(t_predicted_path_200)
E_expected_path_200 = np.load(E_expected_path_200)
E_predicted_path_200 = np.load(E_predicted_path_200)

t_expected_path_300 = f"{base_dir}/CNN_1st/Disc_300/t_E_expected_first_layer_300.npy"
t_predicted_path_300 = f"{base_dir}/CNN_1st/Disc_300/t_E_predicted_first_layer_300.npy"
E_expected_path_300 = f"{base_dir}/CNN_1st/Disc_300/E_expected_first_layer_300.npy"
E_predicted_path_300 = f"{base_dir}/CNN_1st/Disc_300/E_predicted_first_layer_300.npy"
t_expected_path_300 = np.load(t_expected_path_300)
t_predicted_path_300 = np.load(t_predicted_path_300)
E_expected_path_300 = np.load(E_expected_path_300)
E_predicted_path_300 = np.load(E_predicted_path_300)


t_expected_path_400 = f"{base_dir}/CNN_1st/Disc_400/t_E_expected_first_layer_400.npy"
t_predicted_path_400 = f"{base_dir}/CNN_1st/Disc_400/t_E_predicted_first_layer_400.npy"
E_expected_path_400 = f"{base_dir}/CNN_1st/Disc_400/E_expected_first_layer_400.npy"
E_predicted_path_400 = f"{base_dir}/CNN_1st/Disc_400/E_predicted_first_layer_400.npy"

t_expected_path_500 = f"{base_dir}/CNN_1st/Disc_500/t_E_expected_first_layer_500.npy"
t_predicted_path_500 = f"{base_dir}/CNN_1st/Disc_500/t_E_predicted_first_layer_500.npy"
E_expected_path_500 = f"{base_dir}/CNN_1st/Disc_500/E_expected_first_layer_500.npy"
E_predicted_path_500 = f"{base_dir}/CNN_1st/Disc_500/E_predicted_first_layer_500.npy"

t_expected_path_400 = np.load(t_expected_path_400)
t_predicted_path_400 = np.load(t_predicted_path_400)
E_expected_path_400 = np.load(E_expected_path_400)
E_predicted_path_400 = np.load(E_predicted_path_400)

t_expected_path_500 = np.load(t_expected_path_500)
t_predicted_path_500 = np.load(t_predicted_path_500)
E_expected_path_500 = np.load(E_expected_path_500)
E_predicted_path_500 = np.load(E_predicted_path_500)



import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import os

def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E = np.zeros_like(t)
    E[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E

t = np.linspace(0, 30, 251, endpoint=True)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
tau = 5.0

base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model'
E = laminarflow(t, tau)

c_out_full = np.convolve(c_0, E / np.sum(E), mode="full")
t_conv_full = np.linspace(t[0] + t[0], t[-1] + t[-1], len(c_out_full))
valid_indices = t_conv_full <= 30
t_conv = t_conv_full[valid_indices]
c_out = c_out_full[valid_indices]


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
        [r"$E$ / $1$", r"$E$ / $1$"],   [r"$E$ / $1$", r"$E$ / $1$"],[r"$E$ / $1$", r"$E$ / $1$"]
    ])
    

axs[0, 0].plot(
    t,  # Ensure this is also 1D
   E/E.max(),
label=r"$E_{100}(t)$", color=ICIWcolors.KELLYGREEN
)

axs[0, 0].plot(
    t_predicted_path_100-5,  # Ensure this is also 1D
    E_predicted_path_100,
label=r"$\hat{E}_{100}(t)$", color="black", linestyle="--"
)


axs[0, 1].plot(
    t,  # Ensure this is also 1D
    E/E.max(),
    label=r"$E_{200}(t)$", color=ICIWcolors.KELLYGREEN
)

axs[0, 1].plot(
    t_predicted_path_200-5,  # Ensure this is also 1D
    E_predicted_path_200,
    label=r"$\hat{E}_{200}(t)$", color="black", linestyle="--"
)


axs[1, 0].plot(
    t_expected_path_300,  # Ensure this is also 1D
    E_expected_path_300,
    label=r"$E_{300}(t)$", color=ICIWcolors.KELLYGREEN
)

axs[1, 0].plot(
    t_predicted_path_300,  # Ensure this is also 1D
    E_predicted_path_300,
    label=r"$\hat{E}_{300}(t)$", color="black", linestyle="--"
)


axs[1, 1].plot(
    t_expected_path_400,  # Ensure this is also 1D
    E_expected_path_400,
    label=r"$E_{400}(t)$", color=ICIWcolors.KELLYGREEN
)

axs[1, 1].plot(
    t_predicted_path_400,  # Ensure this is also 1D
    E_predicted_path_400,
    label=r"$\hat{E}_{400}(t)$", color="black", linestyle="--"
)



axs[2, 0].plot(
    t,  # Ensure this is also 1D
    E/E.max(),
    label=r"$E_{500}(t)$", color=ICIWcolors.KELLYGREEN
)

axs[2, 0].plot(
    t_predicted_path_500,  # Ensure this is also 1D
    E_predicted_path_500,
    label=r"$\hat{E}_{500}(t)$", color="black", linestyle="--"
)

axs[0, 0].legend(loc="best")
axs[0, 1].legend(loc="best")

axs[1, 0].set_xlim((0, 25)) 
axs[1, 1].set_xlim((0, 25)) 



plt.savefig(os.path.join(base_dir, f"disc.png"), dpi=300)
plt.show()







