import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"
PWD=DATA /"0001"/ "C_001" / "H_085_C1" / "S_009_C1" 

def process_data(file_num):

    x_file_path = PWD / f"TOA_MGA_20231020_009_{file_num:06d}_x.npy"
    t_file_path = PWD / f"TOA_MGA_20231020_009_{file_num:06d}_t.npy"


    x = np.load(x_file_path)
    t = np.load(t_file_path)
    
    # print("Dimensions of x:", x.shape)
    # print("Dimensions of t:", t.shape)

    f = sc.interpolate.interp1d(t, x[0, :])
    t_end_2=t[-71]
    t_end = t[-1]
    t_start = t_end - 41
    print(t_end)
    print(t_start)
    print(t_end_2)
    t_evel = np.linspace(t_start, t_end_2, 200)
    t_pretty = t_evel - t_start
    x_evel = f(t_evel)
    print(t_pretty)


    # f = sc.interpolate.interp1d(t, x[0, :])
    # t_n = t[-1]
    # t_b = t_n - 41
    # t_evel = np.linspace(t_b, t_n, 500)
    # t_pretty = t_evel - t_b
    # x_evel = f(t_evel)


    # np.save(PWD / f"TOA_MGA_20231020_009_{file_num:06d}_t_processed_2nlayer_200_disc.npy", t_pretty)
    # np.save(PWD / f"TOA_MGA_20231020_009_{file_num:06d}_x_processed_2nlayer_200_disc.npy", x_evel)


    plt.plot(t_pretty, x_evel, label=file_num)
    #plt.vlines(1.1, 0,0.05)
    #plt.hlines(0.0505, 0,55)

def main():

    start_file_num = 1
    end_file_num = 20


    for file_num in range(start_file_num, end_file_num + 1):
         #if file_num in [12,7]:
             #continue
    #for file_num in [7,17]:
        process_data(file_num)


    plt.legend()
    plt.xlabel("t/s")
    plt.ylabel("x/1")
    #plt.savefig("S_009_processed_eliminated.png")
    #plt.savefig("S_009_processed.png")
    #plt.savefig("S_009_max.png")
    plt.show()

if __name__ == "__main__":
    main()
