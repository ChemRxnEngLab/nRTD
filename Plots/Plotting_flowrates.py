import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots.figures import Elsevier_Sizes
from ICIW_Plots import make_square_subplots
from ICIW_Plots import cm2inch 
import os as os
base_dir="/Users/tuanaoyuncu/Documents/GitHub/nRTD/Plots"
paths_and_prefixes = [
    (Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1"), "TOA_MGA_20231020_009_"),  # H_085_C1
    (Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2"), "TOA_MGA_20231020_012_"),  # H_085_C2
    (Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_135_C1/S_010_C1"), "TOA_MGA_20231020_010_"),  # H_135_C1
    (Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_135_C2/S_013_C2"), "TOA_MGA_20231020_013_"),  # H_135_C2
    (Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1"), "TOA_MGA_20231013_007_"),  # H_185_C1
    (Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_185_C2/S_014_C2"), "TOA_MGA_20231020_014_")   # H_185_C2
]


def plot_all_files_in_subplots():
    """
    Plot all processed files for each path in one frame with 6 subplots, each corresponding to a specific path.
    Only loads files with '_x_processed_norm.npy' and '_t_processed_norm.npy'.
    """
   
    plt.style.use("ICIWstyle")
    fig = plt.figure(figsize=(Elsevier_Sizes.double_column["in"], 40 * cm2inch))
    axs = make_square_subplots(
        fig=fig,
        ax_width=7 * cm2inch, 
        ax_layout=(3, 2),  
        h_sep=1.3 * cm2inch, 
        v_sep=1.5 * cm2inch,  
        sharex=True,  
        sharey=True, 
        xlabel=[r"$t$ / $s$", r"$t$ / $s$"],  
        ylabel=[r"$C$ / $1$", r"$C$ / $1$",r"$C$ / $1$"]   
    )
    
    

    for i, (path, prefix) in enumerate(paths_and_prefixes):

        t_files = sorted(path.glob(f"{prefix}*t_processed_norm.npy"))
        x_files = sorted(path.glob(f"{prefix}*x_processed_norm.npy"))
        

        ax = axs[i // 2, i % 2]  
        ax.set_xlabel(r"$t$ / $s$")
        ax.set_ylabel(r"$C$ / $1$")


        for t_file_path, x_file_path in zip(t_files, x_files):
            t_pretty = np.load(t_file_path)
            x_evel = np.load(x_file_path)
            ax.plot(t_pretty, x_evel, label=f"{t_file_path.stem}", color=ICIWcolors.CERULEAN)
            ax.set_xlim([0, 15])
            if i % 2 == 0:  # Left side column
                ax.axvline(x=1, color="red", linestyle='--', label="x = 1")
        # Add a vertical line at x=1 for the right column (2nd and 4th row)
            else:  # Right side column
                ax.axvline(x=2, color="red", linestyle='--', label="x = 2")
   

    #plt.savefig(os.path.join(base_dir, f"exp.png"), dpi=300)
    plt.show()

plot_all_files_in_subplots()
