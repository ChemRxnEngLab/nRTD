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
#save_dir="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Plots"
save_dir = r"D:\Tuana\nRTD\Plots"


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
#error_LR_2=[7.6e-9,7.7e-9,7.8e-9,7.5e-9,7.6e-9,7.7e-9,7.53e-9]
error_LR_2=[7.6,7.7,7.8,7.5,7.6,7.7,7.53]
epoch_2=[10000,50000,100000,200000,300000,400000,430000]
error_epoch_2=[8.2e-7,9.8e-8,9.45e-9,7.7e-9,7.5e-9,7.54e-9,7.54e-9]
plt.style.use("ICIWstyle")
fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 15 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(1, 2),  
    h_sep=1.7 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=False,
    sharey=False,
    xlabel=["$Learning\, \,Rate\, \,/\, \,1$","$Epoch\, \,/\, \,1$"], 
    ylabel=
        [["$Test\, \,/\, \,loss$","$Train\, \,/\, \,loss$"]]
    
)
axs[0, 0].text(0.15, 1.05, r'Scale : $\times10^{9}$', 
               transform=axs[0, 0].transAxes,
               horizontalalignment='center',
               verticalalignment='bottom')

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
#axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
#axs[0, 0].set_xlim(min(LR) * 0.5, max(LR) * 2)
plt.savefig(os.path.join(save_dir, f"2nd_hypo.png"), dpi=300)
plt.show()


#####EXP
LR_2=[10,1,1e-1,1e-2,1e-3,1e-4,1e-5]
#error_LR_2=[7.6e-9,7.7e-9,7.8e-9,7.5e-9,7.6e-9,7.7e-9,7.53e-9]
error_LR_2 = [0.0059862, 0.0059867, 0.0005986, 0.00059863, 0.0006000, 0.000624, 0.00073]
epoch_2=[1000,10000,20000,25000,30000,35000]
#error_epoch_2=[0.00044378,0.00042183,0.00042169,0.00042125,0.00042089,0.00042088]
error_epoch_2=[4.4378,4.2183,4.2169,4.2125,4.2089,4.2088]
plt.style.use("ICIWstyle")
fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 15 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(1, 2),  
    h_sep=1.7 * cm2inch,  
    v_sep=0.1 * cm2inch, 
    sharex=False,
    sharey=False,
    xlabel=["$Learning\, \, Rate \, / \, 1$","$Epoch\, \,/\, \,1$"], 
    ylabel=
        [["$Test\, \,/\, \,loss$","$Test\, \,/\, \,loss$"]]
    
)
axs[0, 0].text(1.4, 1.05, r'Scale : $\times10^{4}$', 
               transform=axs[0, 0].transAxes,
               horizontalalignment='center',
               verticalalignment='bottom')

axs[0, 1].plot(
    epoch_2,
    error_epoch_2,'o-',
    color=ICIWcolors.FLAME,
)

#axs[0, 1].set_yscale('log')
#axs[0, 1].set_xscale('log')
axs[0, 0].plot(LR_2, error_LR_2,'o-',color=ICIWcolors.KELLYGREEN)
#axs[0, 1].set_yscale('log')
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')
axs[0, 0].set_xticks(LR)
#axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
#axs[0, 0].set_xlim(min(LR) * 0.5, max(LR) * 2)
plt.savefig(os.path.join(save_dir, f"2nd_hypo_exp.png"), dpi=300)
plt.show()

LR_2=[10,1,1e-1,1e-2,1e-3,1e-4,1e-5]
#error_LR_2=[7.6e-9,7.7e-9,7.8e-9,7.5e-9,7.6e-9,7.7e-9,7.53e-9]
error_LR_2 = [0.0059862, 0.0059867, 0.0005986, 0.00059863, 0.0006000, 0.000624, 0.00073]
epoch_3=[1000,3000,4000,5000,6000,7000,8000,8500]
error_epoch_3=[6.3015,5.7675,5.715,5.6763,5.6514,5.563666,5.56319,5.56319]
plt.style.use("ICIWstyle")
fig = plt.figure( figsize=(Elsevier_Sizes.double_column["in"], 15 * cm2inch))  # Increased figure height for better spacing
axs = make_square_subplots(
    fig=fig,
    ax_width=7 * cm2inch,#dimension of the plots
    ax_layout=(1, 2),  
    h_sep=2.2 * cm2inch,  
    v_sep=1 * cm2inch, 
    sharex=False,
    sharey=False,
    xlabel=["$Learning\, \, Rate \, / \, 1$","$Epoch\, \,/\, \,1$"], 
    ylabel=
        [["$Test\, \,/\, \,loss$","$Test\, \,/\, \,loss$"]]
    
)
axs[0, 0].text(1.5, 1.05, r'Scale : $\times10^{4}$', 
               transform=axs[0, 0].transAxes,
               horizontalalignment='center',
               verticalalignment='bottom')

axs[0, 1].plot(
    epoch_3,
    error_epoch_3,'o-',
    color=ICIWcolors.FLAME,
)

#axs[0, 1].set_yscale('log')
#axs[0, 1].set_xscale('log')
axs[0, 0].plot(LR_2, error_LR_2,'o-',color=ICIWcolors.KELLYGREEN)
#axs[0, 1].set_yscale('log')
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')
axs[0, 0].set_xticks(LR)
#axs[0, 0].set_xticklabels([r'$10^{%d}$' % int(np.log10(x)) if x != 1 else '1' for x in LR])
#axs[0, 0].set_xlim(min(LR) * 0.5, max(LR) * 2)
plt.savefig(os.path.join(save_dir, f"2nd_hypo_exp_n.png"), dpi=300)
plt.show()