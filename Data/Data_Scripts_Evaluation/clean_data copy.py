from pathlib import Path
from MS_Calculation import cal_compostion, calc_calibration
import numpy as np

WD = Path.cwd()
DATA = WD / "Data"


def main():
    folder = WD.parent / "C_001" / "H_185_C1" / "S_007_C1"
    print(folder.is_dir())
    calibration_file = folder / "005_0Ar_010_0He_185_0H2_otherdescription2.txt"
    print(calibration_file.is_file())

    # data_file = folder / "TOA_MGA_20231013_007_000004.txt"
    # print(data_file.is_file())
    # data_filename = data_file.stem

    # RSF, cal_header = calc_calibration([calibration_file])
    # n_t, x_i, t = cal_compostion(data_file, RSF, cal_header)
    # np.save(folder / (data_filename + "_x.npy"), x_i)
    # np.save(folder / (data_filename + "_t.npy"), t)

    RSF, cal_header = calc_calibration([calibration_file])

    for data_file in folder.glob("TOA_MGA_*.txt"):
        data_filename = data_file.stem
        n_t, x_i, t = cal_compostion(data_file, RSF, cal_header)
        np.save(folder / (data_filename + "_x.npy"), x_i)
        np.save(folder / (data_filename + "_t.npy"), t)


if __name__ == "__main__":
    main()