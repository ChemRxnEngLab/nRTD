import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
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
import os
save_dir="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Plots"

file_path = "/Users/tuanaoyuncu/Downloads/wandb_export_2024-12-19T00_13_37.633+01_00.csv"

df=pd.read_csv(file_path,delimiter=",",usecols=[0,4,10,16,22,28,34,40])
print(df.head())


##################################
plt.style.use("ICIWstyle")
fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 25 * cm2inch))  # Increased figure height for better spacing
ax = make_rect_ax(
    fig,
    ax_width=16 * cm2inch,#dimension of the plots
    ax_height=3,
    xlabel="$Epoch/1$", 
    ylabel=
        "$Train/loss$"
    
)
ax.plot(df["epoch"], df["true-pond-1762 - train/loss"],
    label=r"$Learning Rate=10^{-5}$",
    color=ICIWcolors.CRIMSON,
)
ax.plot(df["epoch"], df["mild-salad-1812 - train/loss"],
    label=r"$Learning Rate=10^{-4}$",
    color=ICIWcolors.FLAME,
)
ax.plot(df["epoch"], df["elated-pine-1849 - train/loss"],
    label=r"$Learning Rate=10^{-3}$",color="#4682B4"
    ,
)
ax.plot(df["epoch"], df["brisk-haze-1756 - train/loss"],
    label=r"$Learning Rate=10^{-2}$",
    color=ICIWcolors.KELLYGREEN,
)
ax.plot(df["epoch"], df["mild-universe-1757 - train/loss"],
    label=r"$Learning Rate=10^{-1}$",
    color="#A0CFCF",
)

ax.plot(df["epoch"], df["zany-aardvark-1758 - train/loss"],
    label=r"$Learning Rate=1$",
    color="lightpink",
)
ax.plot(df["epoch"], df["twilight-mountain-1759 - train/loss"],
    label=r"$Learning Rate=10$",
    color=ICIWcolors.DRAB,
)


plt.legend()

ax.set_yscale('log')
ax.text(0.01, 1.075, 'Y-axis: Logarithmic Scale', transform=ax.transAxes,verticalalignment='top')
# axs[0, 1].plot(selected_values, train_loss_values_63,color=ICIWcolors.KELLYGREEN)

#plt.savefig(os.path.join(save_dir, f"dynamic_profiles_1512.png"), dpi=300)
plt.show()

##############################
#LR
LR=[10,1,1e-1,1e-2,1e-3,1e-4,1e-5]
error_2=[7.4647e-8,7.46e-8,7.4541e-8,7.4523e-8,7.4537e-8,1.1904e-7,0.00067261]#7.5e-8
fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 25 * cm2inch))  # Increased figure height for better spacing
ax = make_rect_ax(
    fig,
    ax_width=16 * cm2inch,#dimension of the plots
    ax_height=3,
    xlabel="$Learning \, Rate/1$", 
    ylabel=
        "$Test/loss$"
    
)
ax.plot(LR, error_2,'o-',
    color=ICIWcolors.FLAME
)


ax.set_yscale('log')
ax.set_xscale('log')
ax.text(0.01, 1.075, 'Y-axis and X-axis : Logarithmic Scale', transform=ax.transAxes,verticalalignment='top')
# axs[0, 1].plot(selected_values, train_loss_values_63,color=ICIWcolors.KELLYGREEN)

#plt.savefig(os.path.join(save_dir, f"LR.png"), dpi=300)
plt.show()
##############################2ndlayer

LR_2=[10,1,1e-1,1e-2,1e-3,1e-4,1e-5]
error_LR_2=[7.6,7.7,7.8,7.5,7.6,7.7,7.53]
epoch_2=[10000,50000,100000,200000,300000,400000,430000]
error_epoch_2=[8.2e-7,9.8e-8,9.45e-9,7.7e-9,7.5e-9,7.54e-9,7.54e-9]
plt.style.use("ICIWstyle")
fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 12 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(1, 2),  
    h_sep=1.8 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=False,
    sharey=False,
    xlabel=["$Learning\,Rate/1$","$Epoch/1$"], 
    ylabel=
        [["$Test/loss$","$Train/loss$"]]
    
)

axs[0, 1].plot(
    epoch_2,
    error_epoch_2,'o-',
    color=ICIWcolors.FLAME,
)

axs[0, 1].set_yscale('log')
# axs[0, 1].set_xscale('log')
axs[0, 0].plot(LR_2, error_LR_2,'o-',color=ICIWcolors.KELLYGREEN)
#axs[0, 1].set_yscale('log')
axs[0, 0].set_xscale('log')
#axs[0, 0].set_yscale('log')
axs[0, 0].set_xticks(LR)
axs[0,0].text(0.4, 4.23, 'Scale: $10^{-9}$', transform=ax.transAxes,verticalalignment='top')
#axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
#axs[0, 0].set_xlim(min(LR) * 0.5, max(LR) * 2)
plt.savefig(os.path.join(save_dir, f"2nd_hypo.png"), dpi=300)
plt.show()
# #################
# t10 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1_001/TOA_MGA_20231020_009_000001_t_processed.npy")
# x10 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1_001/TOA_MGA_20231020_009_000001_x_processed.npy")
# x10 /= 224
# x10_p=x10/x10.max()
# t11 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_t_processed.npy")
# x11 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_x_processed.npy")
# x11 /= 224
# x11_p=x11/x11.max()
# t4 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_t_processed.npy")
# t5 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_135_C2/S_013_C2_001/TOA_MGA_20231020_013_000001_t_processed.npy")
# fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 25 * cm2inch))
# ax = make_rect_ax(
#     fig,
#     ax_width=16 * cm2inch,#dimension of the plots
#     ax_height=3,
#     xlabel="$t\,/\,1$", 
#     ylabel=
#         "$Train/loss$"
    
# )
# ax.plot(t4, x10_p, label='(1)', color=ICIWcolors.KELLYGREEN)
# ax.plot(t5, x11_p, label='(2)', color=ICIWcolors.FLAME)
# x_min = min(t4)  # t4'ün minimum değeri
# x_max = max(t4)  # t4'ün maksimum değeri
# xticks = np.arange(x_min, x_max + 1, 2)
# ax.set_xticks(xticks)
# ax.set_xlim(0,20)

# plt.savefig(os.path.join(save_dir, f"exp_positions.png"), dpi=300)
# plt.show()