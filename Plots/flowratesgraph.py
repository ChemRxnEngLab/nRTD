import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ICIW_Plots.colors as ICIWcolors
from ICIW_Plots.figures import Elsevier_Sizes
from ICIW_Plots import make_square_subplots
from ICIW_Plots import cm2inch  # Assuming this is already defined
import os

# Define base directory and Windows paths
base_dir = r"D:\Tuana\nRTD\Plots"

# Define all paths and prefixes (Windows-style paths)
paths_and_prefixes = [
    (Path(r"D:\Tuana\nRTD\Data\0001\C_001\H_085_C1\S_009_C1"), "TOA_MGA_20231020_009_"),  # H_085_C1
    (Path(r"D:\Tuana\nRTD\Data\0001\C_002\H_085_C2\S_012_C2"), "TOA_MGA_20231020_012_"),  # H_085_C2
    (Path(r"D:\Tuana\nRTD\Data\0001\C_001\H_135_C1\S_010_C1"), "TOA_MGA_20231020_010_"),  # H_135_C1
    (Path(r"D:\Tuana\nRTD\Data\0001\C_002\H_135_C2\S_013_C2"), "TOA_MGA_20231020_013_"),  # H_135_C2
    (Path(r"D:\Tuana\nRTD\Data\0001\C_001\H_185_C1\S_007_C1"), "TOA_MGA_20231013_007_"),  # H_185_C1
    (Path(r"D:\Tuana\nRTD\Data\0001\C_002\H_185_C2\S_014_C2"), "TOA_MGA_20231020_014_")   # H_185_C2
]


# Plotting function adapted to your template
def plot_all_files_in_subplots():
    """
    Plot all processed files for each path in one frame with 6 subplots, each corresponding to a specific path.
    Only loads files with '_x_processed_norm.npy' and '_t_processed_norm.npy'.
    """
    # Prepare the main figure using ICIW_Plots styling
    plt.style.use("ICIWstyle")
    fig = plt.figure(figsize=(Elsevier_Sizes.double_column["in"], 40 * cm2inch))  # Increased figure height for better spacing
    axs = make_square_subplots(
        fig=fig,
        ax_width=7 * cm2inch,  # Dimension of the plots
        ax_layout=(3, 2),  # 3 rows, 2 columns
        h_sep=1.2 * cm2inch,  # Horizontal separation between plots
        v_sep=1.5 * cm2inch,  # Vertical separation between plots
        sharex=True,  # Sharing x-axis between subplots
        sharey=True,  # Sharing y-axis between subplots
        xlabel=[r"$t$ / $s$", r"$t$ / $s$"],  # Labels for x-axis (2 columns)
        ylabel=[r"$C$ / $1$", r"$C$ / $1$", r"$C$ / $1$"]   # Labels for y-axis (2 columns)
    )
    legend_labels = [
r"$c_{1,1} = 2.23,molm^{-3}$", # First column, first row
        r"$c_{1,2} = 2.23,molm^{-3}$",  # Second column, first row
        r"$c_{2,1} = 1.49,molm^{-3}$",  # First column, second row
        r"$c_{2,2} = 1.49,molm^{-3}$",  # Second column, second row
        r"$c_{3,1} = 1.12,molm^{-3}$",  # First column, third row
        r"$c_{3,2} = 1.12,molm^{-3}$"   # Second column, third row
    ]

    # Loop through each path and prefix and plot in the corresponding subplot
    for i, (path, prefix) in enumerate(paths_and_prefixes):
        # Find all files that end with the appropriate suffix
        t_files = sorted(path.glob(f"{prefix}*t_processed_norm.npy"))
        x_files = sorted(path.glob(f"{prefix}*x_processed_norm.npy"))
        
        # Get the current axis to plot on
        ax = axs[i // 2, i % 2]  # Positioning based on the index
        ax.set_xlabel(r"$t$ / $s$")
        ax.set_ylabel(r"$C$ / $1$")

        # Plot all files in the current path on the same subplot
        for t_file_path, x_file_path in zip(t_files, x_files):
            t_pretty = np.load(t_file_path)
            x_evel = np.load(x_file_path)
            ax.plot(t_pretty, x_evel, label=f"{t_file_path.stem}", color=ICIWcolors.CERULEAN)
            ax.set_xlim([0, 5])
            ax.legend([legend_labels[i]], loc='best')  # Add the custom legend for the current subplot

            if i % 2 == 0:  # Left side column
                ax.axvline(x=1, color="red", linestyle='--', label="x = 1")
            else:  # Right side column
                ax.axvline(x=2, color="red", linestyle='--', label="x = 2")
        
    # Save the plot
    plt.savefig(os.path.join(base_dir, "exp.png"), dpi=300)
    plt.show()

# Call the function to plot all files
plot_all_files_in_subplots()
