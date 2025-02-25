from pathlib import Path
from MS_Calculation import cal_compostion, calc_calibration
import numpy as np

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"


def main():
    folder = DATA /"0004"/ "C_001" / "H_080_C1" / "S_051_C1"
    print(folder)
    print(folder.is_dir())
    #calibration_file = folder / "005_0Ar_010_0He_185_0H2_otherdescription"

    calib_files = list(folder.glob("*_*Ar_*_*He_*_*H2_*.txt"))

    print("The following calibration files were found:\n")
    for calib_file in calib_files:
        print(calib_file)

    # print(calibration_file.is_file())

    # data_file = folder / "TOA_MGA_20231013_007_000004.txt"
    # print(data_file.is_file())
    # data_filename = data_file.stem

    # RSF, cal_header = calc_calibration([calibration_file])
    # n_t, x_i, t = cal_compostion(data_file, RSF, cal_header)
    # np.save(folder / (data_filename + "_x.npy"), x_i)
    # np.save(folder / (data_filename + "_t.npy"), t)

    RSF, cal_header = calc_calibration(calib_files)

    for data_file in folder.glob("TOA_MGA_*.txt"):
        data_filename = data_file.stem
        n_t, x_i, t = cal_compostion(data_file, RSF, cal_header)
        np.save(folder / (data_filename + "_x.npy"), x_i)
        np.save(folder / (data_filename + "_t.npy"), t)


if __name__ == "__main__":
    main()
