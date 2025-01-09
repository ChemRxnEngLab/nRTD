import matplotlib.pyplot as plt
import sys
import os
import torch
import numpy as np
module_path = r"D:\Tuana\nRTD\lib"
sys.path.append(module_path)
import datetime
from sympy import ceiling
from ICIW_Plots import make_square_ax, cm2inch
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots import make_square_subplots
from ICIW_Plots.figures import Elsevier_Sizes
save_dir = r"D:\Tuana\nRTD\Plots"
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
save_dir = r"D:\Tuana\nRTD\Plots"
file_path = r"D:\Tuana\nRTD\Plots\wandb_export_2024-12-27T01_42_23.467+01_00.csv"
#file_path="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Plots/wandb_export_2024-12-27T01_42_23.467+01_00.csv"
#save_dir="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Plots"

df=pd.read_csv(file_path,delimiter=",",usecols=[0,4,10])
print(df.head())

#epoch,"Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000_RTD2 - _step","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000_RTD2 - _step__MIN","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000_RTD2 - _step__MAX","","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000_RTD2 - train/loss__MIN","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000_RTD2 - train/loss__MAX","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000 - _step","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000 - _step__MIN","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000 - _step__MAX","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000 - train/loss","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000 - train/loss__MIN","Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000 - train/loss__MAX"
##################################
plt.style.use("ICIWstyle")
fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 25 * cm2inch))  # Increased figure height for better spacing
ax = make_rect_ax(
    fig,
    ax_width=15.5 * cm2inch,#dimension of the plots
    ax_height=3,
    xlabel="$Epoch/1$", 
    ylabel=
        "$Train/loss$"
    
)
ax.plot(df["epoch"], df["Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000_RTD2 - train/loss"],
    label=r"$Case:1$",
    color=ICIWcolors.KELLYGREEN,
)

ax.plot(df["epoch"], df["Layer_2_CH_n_1_out_200_LC_learning_rate_0.01_2nd_epoch_900000 - train/loss"],
    label=r"$Case:2$",
    color=ICIWcolors.FLAME,
)

plt.legend()

ax.set_yscale('log')
#ax.text(0.01, 1.075, 'Y-axis: Logarithmic Scale', transform=ax.transAxes,verticalalignment='top')
# axs[0, 1].plot(selected_values, train_loss_values_63,color=ICIWcolors.KELLYGREEN)

plt.savefig(os.path.join(save_dir, f"dynamic_profiles_2nlayer_0801.png"), dpi=300)
plt.show()