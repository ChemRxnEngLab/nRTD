# Calculate the RSF matrix for given MS data

import numpy as np
from tkinter import filedialog
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy.typing as npt

## Bestimmung der Konzentrationen aus Kaibrierdatei
comp = np.array(["time", "Ar", "He", "H2"])
# indx =        [0,     1,    2,   3] # component indices


class Calibration:
    # Ionisierungswahrscheinlichkeiten aus der Literatur
    PI = {
        "He": 0.14,
        "Ar": 1.2,
        "H2": 0.44,
    }

    def __init__(
        self,
        components: list[str],
        cal_files: list[Path],
        reference_ion="He",
    ):
        self.cal_files = cal_files
        self.components = components
        self.reference_ion = reference_ion
        self._ion_ref = self.PI[reference_ion]
        self.RSF = self.calc_RSF()

    @property
    def ion_ref(self):
        return self._ion_ref

    @ion_ref.setter
    def ion_ref(self, ion: str):
        self._ion_ref = self.PI["ion"]

    def get_xi_ref(self, path_cal):
        # template for titling the ms calibration files
        # 000_0Ar_000_0He_000_0H2_otherdescription.txt
        # not included components can be omitted
        filename = os.path.basename(path_cal)
        Vdot = np.zeros_like(self.components, dtype=float)
        print("\nVolume flows during calibration:")
        for i, comp in enumerate(self.components):
            ind_i = filename.find(
                comp + "_"
            )  # search for the component i in the filename
            if (
                ind_i != -1
            ):  # if ind_i equals -1, the component is not included in the file
                if filename[ind_i - 2] == "_":  # one dezimal given
                    Vdot[i] = (
                        float(filename[ind_i - 5 : ind_i - 2])
                        + float(filename[ind_i - 1]) / 10
                    )  # read the volumeflows from the filename
                    print(comp[i], ":", Vdot[i])
                elif filename[ind_i - 3] == "_":  # two dezimals given
                    Vdot[i] = (
                        float(filename[ind_i - 6 : ind_i - 3])
                        + float(filename[ind_i - 2 : ind_i]) / 100
                    )  # read the volumeflows from the filename
                    print(comp, ":", Vdot[i])

        x_i_cal = Vdot / sum(Vdot)  # mol/mol
        return x_i_cal

    def _calc_RSF_all(self) -> npt.NDArray:
        RSF_all = np.array([])
        for path_cal in self.cal_files:
            print("Used calibration file:", path_cal)
            ms_header_cal = np.loadtxt(
                path_cal,
                delimiter="\t",
                skiprows=2,
                usecols=(1, 3, 5),
                dtype=str,
                max_rows=1,
            )
            self.ms_header_cal = np.char.replace(ms_header_cal, '"', "")

            ms_data_cal = np.loadtxt(
                path_cal,
                delimiter="\t",
                skiprows=3,
                usecols=(1, 3, 5),
            )
            ms_data_avg_cal = np.mean(ms_data_cal, axis=0)
            ms_data_std_cal = np.std(ms_data_cal, axis=0)

            if len(self.ms_header_cal) != len(ms_data_avg_cal):
                raise ValueError("Header and data don't fit!")

            x_i_cal = self.get_xi_ref(path_cal)

            # Choose reference He
            x_ref = x_i_cal[self.components == "He"]  # x_ref = x_He as reference
            Idot_ref = ms_data_avg_cal[ms_header_cal == "Mass 4"]  # Idot_ref = Mass 4

            # Choose where to evaluate the RSF values
            eval_at = np.zeros(
                (len(self.components), len(ms_data_avg_cal))
            )  # if eval_at is one at a position this combination is evaluated regarding the RSF(component,mass)
            # eval_at[np.where(comp=='Ar')[0][0],np.where(ms_header_cal=='Mass 20')[0][0]] = 1 # Ar,20
            eval_at[self.components == "Ar", ms_header_cal == "Mass 40"] = 1  # Ar,40
            eval_at[self.components == "He", ms_header_cal == "Mass 4"] = 1  # He,4
            eval_at[self.components == "H2", ms_header_cal == "Mass 2"] = 1  # H2,2

            # calculate RSF matrix
            RSF_temp = np.zeros_like(eval_at)
            for mm in range(eval_at.shape[1]):
                for i in range(len(x_i_cal)):
                    if (
                        x_i_cal[i] != 0
                    ):  # only calculate RSF if component is present in the calibration measurement
                        RSF_temp[i, mm] = (
                            x_ref
                            / x_i_cal[i]
                            * ms_data_avg_cal[mm]
                            / Idot_ref
                            * eval_at[i, mm]
                        )
            if path_cal == self.cal_files[0]:
                RSF_all = np.array([RSF_temp]).copy()
            else:
                RSF_all = np.concatenate((RSF_all, np.array([RSF_temp])), axis=0)
        return RSF_all

    def calc_RSF(self) -> npt.NDArray:
        # Calculate final RSF matrix
        # Take the mean of values that occur more than one time

        RSF_all = self._calc_RSF_all()

        RSF = np.true_divide(RSF_all.sum(0), (RSF_all != 0).sum(0))
        RSF = np.nan_to_num(RSF, nan=0.0, posinf=0.0, neginf=0.0)

        # Use the literature values for all non calibrated components
        # Reference: Cracking Patterns
        non_calib = ~RSF.any(axis=1)
        # non_calib = np.full((len(comp)),True) # uncomment to show the literatur data
        if non_calib[self.components == "Ar"]:  # Ar not calibrated
            RSF[self.components == "Ar", self.ms_header_cal == "Mass 20"] = (
                1 * self.PI["Ar"] / self.ion_ref
            )  # Ar,20
            RSF[self.components == "Ar", self.ms_header_cal == "Mass 40"] = (
                0.1 * self.PI["Ar"] / self.ion_ref
            )  # Ar,40
        if non_calib[self.components == "He"]:  # He not calibrated
            RSF[self.components == "He", self.ms_header_cal == "Mass 4"] = (
                1 * self.PI["He"] / self.ion_ref
            )
        if non_calib[self.components == "H2"]:  # H2 not calibrated
            RSF[self.components == "H2", self.ms_header_cal == "Mass 2"] = (
                1 * self.PI["H2"] / self.ion_ref
            )

        print()
        print("RSF final in %:\n", np.array(RSF * 100, dtype=int), "\n")
        print("RSF matrix calculation finished!")
        return RSF


class MSData:
    def __init__(
        self,
        path_exp: Path,
        calib: Calibration,
    ):
        self.path_exp = path_exp
        self.calib = calib

    @property
    def ms_time(self) -> npt.NDArray:
        _t = np.loadtxt(
            self.path_exp,
            delimiter="\t",
            skiprows=3,
            usecols=(0),
            dtype=str,
        )
        _t = pd.to_datetime(_t[:], format='"%d.%m.%Y %H:%M:%S.%f"')
        _t = np.array(_t, dtype=np.datetime64)
        return _t

    @property
    def ms_time_elap(self) -> npt.NDArray:
        t = self.ms_time
        _t = (t - t[0]) / np.timedelta64(1, "s")
        return _t

    def cal_amount(self):
        print("Calculating the compositions of the given experiment...")

        ms_header_exp = np.loadtxt(
            self.path_exp,
            delimiter="\t",
            skiprows=2,
            usecols=(1, 2, 3, 4, 5, 6),
            dtype=str,
            max_rows=1,
        )
        ms_header_exp = np.char.replace(ms_header_exp, '"', "")

        ms_data_exp = np.loadtxt(
            self.path_exp,
            delimiter="\t",
            skiprows=3,
            usecols=(1, 2, 3, 4, 5, 6),
        )

        # Caclulate each component
        # Argon
        nt_Ar = (
            ms_data_exp[:, ms_header_exp == "Mass 40"]
            / self.calib.RSF[
                self.calib.components == "Ar", self.calib.ms_header_cal == "Mass 40"
            ]
        )
        # Helium
        nt_He = (
            ms_data_exp[:, ms_header_exp == "Mass 4"]
            / self.calib.RSF[
                self.calib.components == "He", self.calib.ms_header_cal == "Mass 4"
            ]
        )
        # Hydrogen
        nt_H2 = (
            ms_data_exp[:, ms_header_exp == "Mass 2"]
            / self.calib.RSF[
                self.calib.components == "H2", self.calib.ms_header_cal == "Mass 2"
            ]
        )

        nt_all = np.array([nt_Ar, nt_He, nt_H2])

        return nt_all

    def cal_composition(self):
        n = self.cal_amount()
        x_i = n / np.sum(n, axis=0)
        return x_i


def calc_calibration(cal_files):
    for path_cal in cal_files:
        print("Used calibraion file:", path_cal)
        ms_header_cal = np.loadtxt(
            path_cal,
            delimiter="\t",
            skiprows=2,
            usecols=(1, 3, 5),
            dtype=str,
            max_rows=1,
        )
        ms_header_cal = np.char.replace(ms_header_cal, '"', "")
        ms_data_cal = np.loadtxt(
            path_cal, delimiter="\t", skiprows=3, usecols=(1, 3, 5)
        )
        ms_data_avg_cal = np.mean(ms_data_cal, axis=0)
        ms_data_std_cal = np.std(ms_data_cal, axis=0)
        if len(ms_header_cal) != len(ms_data_avg_cal):
            raise ValueError("Header and data don´t fit!")

        # template for titling the ms calibration files
        # 000_0Ar_000_0He_000_0H2_otherdescription.txt
        # not included components can be omitted
        filename = os.path.basename(path_cal)
        Vdot = np.zeros_like(comp, dtype=float)
        print("\nVolume flows during calibration:")
        for i in range(len(comp)):
            ind_i = filename.find(
                comp[i] + "_"
            )  # search for the component i in the filename
            if (
                ind_i != -1
            ):  # if ind_i equals -1, the component is not included in the file
                if filename[ind_i - 2] == "_":  # one dezimal given
                    Vdot[i] = (
                        float(filename[ind_i - 5 : ind_i - 2])
                        + float(filename[ind_i - 1]) / 10
                    )  # read the volumeflows from the filename
                    print(comp[i], ":", Vdot[i])
                elif filename[ind_i - 3] == "_":  # two dezimals given
                    Vdot[i] = (
                        float(filename[ind_i - 6 : ind_i - 3])
                        + float(filename[ind_i - 2 : ind_i]) / 100
                    )  # read the volumeflows from the filename
                    print(comp[i], ":", Vdot[i])

        x_i_cal = Vdot / sum(Vdot)  # mol/mol

        # Ionisierungswahrscheinlichkeiten aus der Literatur
        He_ion = 0.14
        Ar_ion = 1.2
        H2_ion = 0.44

        # Choose reference
        print("\nHe as reference")
        x_ref = x_i_cal[np.where(comp == "He")[0][0]]  # x_ref = x_He as reference
        Idot_ref = ms_data_avg_cal[
            np.where(ms_header_cal == "Mass 4")[0][0]
        ]  # Idot_ref = Mass 4
        ion_ref = He_ion

        # Choose where to evaluate the RSF values
        eval_at = np.zeros(
            (len(comp), len(ms_data_avg_cal))
        )  # if eval_at is one at a position this combination is evaluated regarding the RSF(component,mass)
        # eval_at[np.where(comp=='Ar')[0][0],np.where(ms_header_cal=='Mass 20')[0][0]] = 1 # Ar,20
        eval_at[
            np.where(comp == "Ar")[0][0], np.where(ms_header_cal == "Mass 40")[0][0]
        ] = 1  # Ar,40
        eval_at[
            np.where(comp == "He")[0][0], np.where(ms_header_cal == "Mass 4")[0][0]
        ] = 1  # He,4
        eval_at[
            np.where(comp == "H2")[0][0], np.where(ms_header_cal == "Mass 2")[0][0]
        ] = 1  # H2,2

        # calculate RSF matrix
        RSF_temp = np.zeros_like(eval_at)
        for mm in range(eval_at.shape[1]):
            for i in range(len(x_i_cal)):
                if (
                    x_i_cal[i] != 0
                ):  # only calculate RSF if component is present in the calibration measurement
                    RSF_temp[i, mm] = (
                        x_ref
                        / x_i_cal[i]
                        * ms_data_avg_cal[mm]
                        / Idot_ref
                        * eval_at[i, mm]
                    )

        if path_cal == cal_files[0]:
            RSF_all = np.array([RSF_temp])
        else:
            RSF_all = np.concatenate((RSF_all, np.array([RSF_temp])), axis=0)

    # Calculate final RSF matrix
    # Take the mean of values that occur more than one time

    RSF = np.true_divide(RSF_all.sum(0), (RSF_all != 0).sum(0))
    RSF = np.nan_to_num(RSF, nan=0.0, posinf=0.0, neginf=0.0)

    # Use the literature values for all non calibrated components
    # Reference: Cracking Patterns
    non_calib = ~RSF.any(axis=1)
    # non_calib = np.full((len(comp)),True) # uncomment to show the literatur data
    if non_calib[np.where(comp == "Ar")[0][0]]:  # Ar not calibrated
        RSF[
            np.where(comp == "Ar")[0][0], np.where(ms_header_cal == "Mass 20")[0][0]
        ] = (
            1 * Ar_ion / ion_ref
        )  # Ar,20
        RSF[
            np.where(comp == "Ar")[0][0], np.where(ms_header_cal == "Mass 40")[0][0]
        ] = (
            0.1 * Ar_ion / ion_ref
        )  # Ar,40
    if non_calib[np.where(comp == "He")[0][0]]:  # He not calibrated
        RSF[np.where(comp == "He")[0][0], np.where(ms_header_cal == "Mass 4")[0][0]] = (
            1 * He_ion / ion_ref
        )
    if non_calib[np.where(comp == "H2")[0][0]]:  # H2 not calibrated
        RSF[np.where(comp == "H2")[0][0], np.where(ms_header_cal == "Mass 2")[0][0]] = (
            1 * H2_ion / ion_ref
        )

    print()
    print("RSF final in %:\n", np.array(RSF * 100, dtype=int), "\n")
    print("RSF matrix calculation finished!")
    return (RSF, ms_header_cal)


def cal_compostion(path_exp, RSF, ms_header_cal):
    print("Calculating the compositions of the given experiment...")

    ms_header_exp = np.loadtxt(
        path_exp,
        delimiter="\t",
        skiprows=2,
        usecols=(1, 2, 3, 4, 5, 6),
        dtype=str,
        max_rows=1,
    )
    ms_header_exp = np.char.replace(ms_header_exp, '"', "")
    ms_data_exp = np.loadtxt(
        path_exp, delimiter="\t", skiprows=3, usecols=(1, 2, 3, 4, 5, 6)
    )
    # read the measurement time
    print("Reading the measurement time...")
    ms_time_exp = np.loadtxt(
        path_exp, delimiter="\t", skiprows=3, usecols=(0), dtype=str
    )

    # extract evaluated times
    pd_times = pd.to_datetime(ms_time_exp[:], format='"%d.%m.%Y %H:%M:%S.%f"')
    ms_times = np.array(pd_times, dtype=np.datetime64)
    ms_time_elap = (ms_times - ms_times[0]) / np.timedelta64(1, "s")

    # Caclulate each component
    # Argon
    nt_Ar = (
        ms_data_exp[:, np.where(ms_header_exp == "Mass 40")[0][0]]
        / RSF[np.where(comp == "Ar")[0][0], np.where(ms_header_cal == "Mass 40")[0][0]]
    )
    # Helium
    nt_He = (
        ms_data_exp[:, np.where(ms_header_exp == "Mass 4")[0][0]]
        / RSF[np.where(comp == "He")[0][0], np.where(ms_header_cal == "Mass 4")[0][0]]
    )
    # Hydrogen
    nt_H2 = (
        ms_data_exp[:, np.where(ms_header_exp == "Mass 2")[0][0]]
        / RSF[np.where(comp == "H2")[0][0], np.where(ms_header_cal == "Mass 2")[0][0]]
    )

    nt_all = np.array(
        [nt_Ar, nt_He, nt_H2]
    )  # ,nt_CO,nt_CO2,nt_CH4,nt_H2O,nt_C2,nt_C3,nt_Ne,nt_C4,nt_C5,nt_C6])
    # Composition
    x_i = nt_all / np.sum(nt_all, axis=0)

    print("Composition calculated!")
    return (nt_all, x_i, ms_time_elap)


def main():
    print("Calculating the RSF matrix...")
    input_filepaths = filedialog.askopenfilenames(
        initialdir=r"/Users/tuanaoyuncu/Desktop/RI_Data_ExpSet1",
        title="Select Calibration Files",
    )

    # calculate the composition
    path_exp = filedialog.askopenfilename(
        initialdir=r"C:\Users\MaxGäßler\Documents\Rohdaten\MS", title="Select MS File"
    )
    print(path_exp)

    RSF, ms_header_cal = calc_calibration(input_filepaths)
    nt_all, x_i, ms_time_elap = cal_compostion(path_exp, RSF, ms_header_cal)

    plt.figure()
    plt.plot(ms_time_elap, x_i[1, :])
    plt.plot(ms_time_elap, x_i[0, :])
    plt.ylim((0, 0.1))
    np.save(path_exp[:-4] + "_x.npy", x_i)
    np.save(path_exp[:-4] + "_t.npy", ms_time_elap)

    print("Script finished!")


if __name__ == "__main__":
    main()
