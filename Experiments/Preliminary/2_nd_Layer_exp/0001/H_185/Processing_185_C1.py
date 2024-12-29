import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"
PWD=DATA /"0001"/ "C_001" / "H_185_C1" / "S_007_C1"

def process_data(file_num):
    x_file_path = PWD / f"TOA_MGA_20231013_007_{file_num:06d}_x.npy"
    t_file_path = PWD / f"TOA_MGA_20231013_007_{file_num:06d}_t.npy"
    x = np.load(x_file_path)
    t = np.load(t_file_path)
    f = sc.interpolate.interp1d(t, x[0, :])
    t_n = t[-1]
    t_b = t_n - 41
    t_evel = np.linspace(t_b, t_n, 200)
    t_pretty = t_evel - t_b
    x_evel_1 = f(t_evel)
    x_evel_2=(x_evel_1/22.4)*1000
    x_evel_2/= x_evel_2.max()
    x_evel=x_evel_2
    

    np.save(PWD / f"TOA_MGA_20231013_007_{file_num:06d}_t_processed_norm_1.npy", t_pretty)
    np.save(PWD / f"TOA_MGA_20231013_007_{file_num:06d}_x_processed_norm_1.npy", x_evel)

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    plt.plot(t_pretty, x_evel, label=file_num)
    #plt.vlines(1.1, 0,0.03)
    #plt.hlines(0.0252, 0,55)

def main():
    start_file_num = 1
    end_file_num = 20
    for file_num in range(start_file_num, end_file_num + 1):
    #for file_num in [9,19]:
        process_data(file_num)

    # Show the legend and plot
    #plt.legend()
    plt.xlabel("t/s")
    plt.ylabel("x/1")
    plt.savefig("s_007_noice.png")
    #plt.savefig("s_007_max.png")
    plt.show()

if __name__ == "__main__":
    main()
