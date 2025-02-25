from pathlib import Path
from MS_Calculation_005 import cal_composition,calc_calibration
import numpy as np

# Define the absolute path to the data folder
data_folder = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data")

def main():
    # Define the folder path relative to the data folder
    folder = data_folder / "0003" / "C_004_0003" / "H_180_C4_0003" / "S_040" 
    print(folder.is_dir())
    
    # Define the absolute path to the calibration file
    calibration_file = data_folder / "0003" / "C_004_0003" / "H_180_C4_0003" / "S_040" / "010_0Ar_010_0He_180_0H2_otherdescription.asc"
    print(calibration_file.is_file())
    
    # Print if the script is taking the specified calibration file
    if calibration_file.is_file():
        print("Taking the specified calibration file:", calibration_file)

    RSF, cal_header = calc_calibration([calibration_file])

    for data_file in folder.glob("TOA_MGA_*.asc"):
        data_filename = data_file.stem
        n_t, x_i, t = cal_composition(data_file, RSF, cal_header)
        np.save(folder / (data_filename + "_x.npy"), x_i)
        np.save(folder / (data_filename + "_t.npy"), t)

    # print("Dimensions of x:", x_i.shape)
    # print("Dimensions of t:", t.shape)


if __name__ == "__main__":
    main()
    
    
    