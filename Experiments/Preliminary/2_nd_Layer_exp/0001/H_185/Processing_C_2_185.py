import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"
PWD=DATA / "0001"/ "C_002" / "H_185_C2" / "S_014_C2" 

def process_data(file_num):
    # Construct file paths for _x.npy and _t.npy
    x_file_path = PWD / f"TOA_MGA_20231020_014_{file_num:06d}_x.npy"
    t_file_path = PWD / f"TOA_MGA_20231020_014_{file_num:06d}_t.npy"

    # Load _x.npy and _t.npy
    x = np.load(x_file_path)
    t = np.load(t_file_path)
    print(t)
    # Interpolate data
    f = sc.interpolate.interp1d(t, x[0, :])
    t_n = t[-1]
    t_b = t_n - 51
    t_evel = np.linspace(t_b, t_n, 485)
    t_pretty = t_evel - t_b
    x_evel = f(t_evel)
    #print(x_evel.shape)

    # Save processed data
    #np.save(PWD / f"TOA_MGA_20231020_014_{file_num:06d}_t_processed.npy", t_pretty)
    #np.save(PWD / f"TOA_MGA_20231020_014_{file_num:06d}_x_processed.npy", x_evel)

    # Plot results
    plt.plot(t_pretty, x_evel, label=file_num)
    plt.vlines(1.8, 0,0.03)
    plt.hlines(0.0255, 0,60)

def main():
    # Specify the range of file numbers you want to process
    start_file_num = 1
    end_file_num = 20

    # Iterate over the file numbers and process each data file
    for file_num in range(start_file_num, end_file_num + 1):
    #for file_num in [19]:
        #if file_num in [19]:
         #  continue
        process_data(file_num)

    # Show the legend and plot
    plt.legend()
    plt.xlabel("t/s")
    plt.ylabel("x/1")
    #plt.savefig("s_014_max.png")
    #plt.savefig("s_014_processed.png")
    plt.show()

if __name__ == "__main__":
    main()
