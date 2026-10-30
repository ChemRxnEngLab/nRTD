import pandas as pd
import numpy as np
import os
import json

import scipy.integrate as spi
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
plt.style.use("ICIWstyle")
import ICIW_Plots
from ICIW_Plots import cm2inch as cm2inch
from ICIW_Plots import make_square_ax
from ICIW_Plots import make_square_subplots
from datetime import datetime, timedelta

class TPREval:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text_file_path = file_path.replace('.XLS', '.txt')
        self.result_folder = os.path.join(os.path.dirname(file_path), "result")
        self.data_array = None

    # def convert_xls_to_text(self):
    #     # Read the XLS file
    #     xls_data = pd.read_excel(self.file_path)

    #     # Save the data to a text file
    #     xls_data.to_csv(self.text_file_path, sep='\t', index=False)

    #     # Replace all commas with dots in the text file
    #     with open(self.text_file_path, 'r') as file:
    #         content = file.read()

    #     content = content.replace(',', '.')
    #     content = content.replace('1.000.0', '1000.0')
    #     content = content.replace('1.000.1', '1000.1')
        


    #     with open(self.text_file_path, 'w') as file:
    #         file.write(content)

    def convert_xls_to_text(self):
    # Read the XLS file
        xls_data = pd.read_excel(self.file_path)

        # Save the data to a text file
        xls_data.to_csv(self.text_file_path, sep='\t', index=False)

        # Read text file
        with open(self.text_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        processed_lines = []

        for i, line in enumerate(lines):
            # Replace commas with dots
            line = line.replace(',', '.')

            # Skip header line (pandas writes header by default)
            if i == 0:
                processed_lines.append(line)
                continue

            parts = line.strip().split('\t')

            if parts and '.' in parts[0]:
                # remove only the FIRST dot in the first column
                parts[0] = parts[0].replace('.', '')

            processed_lines.append('\t'.join(parts) + '\n')

        # Write back to file
        with open(self.text_file_path, 'w', encoding='utf-8') as file:
            file.writelines(processed_lines)


    def extract_data_array(self, header="Time (min)"):
        # data_array = []
        # header_found = False
        # header_line_index = -1

        # with open(self.text_file_path, 'r') as file:
        #     lines = file.readlines()

        #     for index, line in enumerate(lines):
        #         if header_found:
        #             if "Sample Log" in line:
        #                 break
        #             data_array.append([float(value) if value.replace('.', '', 1).isdigit() else value for value in line.strip().split('\t')])
        #         elif header in line:
        #             header_found = True
        #             header_line_index = index

        # # Filter out rows that are not homogeneous
        # data_array = [row for row in data_array if all(isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().replace('.', '', 1).isdigit()) for value in row)]
        # # Ensure all rows have the same length as the header
        # data_array = [row for row in data_array if len(row) == 4]
        # self.data_array = np.array(data_array, dtype=float)

        # print("Header found at line index:", header_line_index)

        with open(self.text_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Start- und Endmarkierungen definieren
        start_marker = "Time (min)	Temperature (°C)	Signal (a.u.)	Injection Event"
        end_marker = "Sample Log"
        
        # Zeilen zwischen den Markierungen extrahieren
        inside = False
        data_lines = []
        for line in lines:
            line = line.strip()
            if line == start_marker:
                inside = True
                continue
            elif line == end_marker:
                break
            if inside:
                data_lines.append(line)

        print(f"Extracted {len(data_lines)} lines of data between markers.")
        # In DataFrame umwandeln

        self.data_df = pd.DataFrame([x.split('\t') for x in data_lines])
        # Konvertieren der Spalten in numerische Werte, Fehlerbehandlung für nicht-numerische Werte
        df_numeric = self.data_df.apply(pd.to_numeric, errors='coerce').dropna()
        self.data_array = df_numeric.to_numpy(dtype=float)

        self.data_array[:,0] = self.data_array[:,0] /1e4  # Zeit in min umrechnen - komma verschiebung 

    def read_sample_mass_and_normalize(self):
        with open(self.text_file_path, "r", encoding="utf-8") as file:
            for line in file:
                if "Sample mass:" in line:
                    # split at colon and take the value part
                    value_part = line.split("Sample mass:")[1].strip()
                    # extract numeric value
                    self.mass = float(value_part.split()[0])
                    self.norm_signal = self.data_array[:, 2] / self.mass

        

    def save_data_array_as_json(self):
        # Nur Dateiname ohne Erweiterung
        filename = os.path.splitext(os.path.basename(self.file_path))[0]  # → "messung_01"
        os.makedirs(self.result_folder, exist_ok=True)
        json_file_path = os.path.join(self.result_folder, f"{filename}_data_array.json")

        with open(json_file_path, 'w') as json_file:
            json.dump(self.data_array.tolist(), json_file)

        print(f"Data array saved as JSON at: {json_file_path}")



    def baseline_and_plot (self, t_start_interpol, t_end_interpol, t_start_interpol_peak, t_end_interpol_peak, x_lim_TPR_low, x_lim_TPR_high, y_lim_TPR_low, y_lim_TPR_high, y_lim_peak_low, y_lim_peak_high, x_lim_peak_low, x_lim_peak_high):
        self.t_start_interpol = t_start_interpol
        self.t_end_interpol = t_end_interpol
        self.start_interpol_peak = t_start_interpol_peak
        self.end_interpol_peak = t_end_interpol_peak
        self.x_lim_TPR_low = x_lim_TPR_low
        self.x_lim_TPR_high = x_lim_TPR_high
        self.y_lim_TPR_low = y_lim_TPR_low
        self.y_lim_TPR_high = y_lim_TPR_high
        self.y_lim_peak_low = y_lim_peak_low
        self.y_lim_peak_high = y_lim_peak_high
        self.x_lim_peak_low = x_lim_peak_low
        self.x_lim_peak_high = x_lim_peak_high

        # set the baseline and plot T over t; signal over t and the peak

        # set the base line
        # Create interpolation function from the first TPR
        start_interpol = np.where(self.data_array[:, 0] > t_start_interpol)[0][0]
        end_interpol = np.where(self.data_array[:, 0] > t_end_interpol)[0][0]
        start_interpol_peak = np.where(self.data_array[:, 0] > t_start_interpol_peak)[0][0]
        end_interpol_peak = np.where(self.data_array[:, 0] > t_end_interpol_peak)[0][0]
        self.baseline_interp = interp1d(np.array((self.data_array[start_interpol,0], self.data_array[end_interpol, 0])), np.array((self.data_array[start_interpol,2], self.data_array[end_interpol,2])), kind='linear', fill_value='extrapolate')
        self.baseline_peak = interp1d(np.array((self.data_array[start_interpol_peak,0], self.data_array[end_interpol_peak, 0])), np.array((self.data_array[start_interpol_peak,2], self.data_array[end_interpol_peak,2])), kind='linear', fill_value='extrapolate')



        fig = plt.figure(figsize = (16*ICIW_Plots.cm2inch,13*ICIW_Plots.cm2inch))

        ax = make_square_subplots(
            fig=fig,
            ax_width=5 * ICIW_Plots.cm2inch,
            ax_layout=(1, 1),
            h_sep=2 * ICIW_Plots.cm2inch,
            v_sep=1 * ICIW_Plots.cm2inch,
            sharex=False,  # share the axis
            sharey=False,  # share the axis
            sharelabel=False,
            xlabel=[r'$t\,/\,\mathrm{min}$'],  # every x axis has the same label
            ylabel=[[r'$T\,/\,\mathrm{°C}$']]#, r'$S^{\mathrm{CO}}_{x=L}\,/\,\mathrm{1}$'],  # every y axis along the rows has the same label
        )
        ax[0,0].plot(self.data_array[:, 0], self.data_array[:, 1], label='TPR')
        #ax[0,0].plot(data.data_array[:, 0], baseline_interp(data.data_array[:, 0]), linestyle='--', color='k', label='baseline')
        plt.legend()
        plt.show()

        fig = plt.figure(figsize = (16*ICIW_Plots.cm2inch,13*ICIW_Plots.cm2inch))

        ax = make_square_subplots(
            fig=fig,
            ax_width=5 * ICIW_Plots.cm2inch,
            ax_layout=(1, 1),
            h_sep=2 * ICIW_Plots.cm2inch,
            v_sep=1 * ICIW_Plots.cm2inch,
            sharex=False,  # share the axis
            sharey=False,  # share the axis
            sharelabel=False,
            xlabel=[r'$t\,/\,\mathrm{min}$'],  # every x axis has the same label
            ylabel=[[r'$\mathrm{Signal}\,/\,\mathrm{a.u.}$']]#, r'$S^{\mathrm{CO}}_{x=L}\,/\,\mathrm{1}$'],  # every y axis along the rows has the same label
        )
        ax[0,0].plot(self.data_array[:, 0], self.data_array[:, 2], label='TPR')
        ax[0,0].plot(self.data_array[:, 0], self.baseline_interp(self.data_array[:, 0]), linestyle='--', color='k', label='baseline')
        ax[0,0].set_xlim(x_lim_TPR_low,x_lim_TPR_high)
        ax[0,0].set_ylim(y_lim_TPR_low,y_lim_TPR_high)
        plt.legend()
        plt.show()

        fig = plt.figure(figsize = (16*ICIW_Plots.cm2inch,13*ICIW_Plots.cm2inch))

        ax = make_square_subplots(
            fig=fig,
            ax_width=5 * ICIW_Plots.cm2inch,
            ax_layout=(1, 1),
            h_sep=2 * ICIW_Plots.cm2inch,
            v_sep=1 * ICIW_Plots.cm2inch,
            sharex=False,  # share the axis
            sharey=False,  # share the axis
            sharelabel=False,
            xlabel=[r'$t\,/\,\mathrm{min}$'],  # every x axis has the same label
            ylabel=[[r'$\mathrm{Signal}\,/\,\mathrm{a.u.}$']]#, r'$S^{\mathrm{CO}}_{x=L}\,/\,\mathrm{1}$'],  # every y axis along the rows has the same label
        )
        ax[0,0].plot(self.data_array[:, 0], self.data_array[:, 2], label='TPR')
        ax[0,0].plot(self.data_array[:, 0], self.baseline_peak(self.data_array[:, 0]), linestyle='--', color='k', label='baseline')
        ax[0,0].set_ylim(y_lim_peak_low,y_lim_peak_high)
        ax[0,0].set_xlim(x_lim_peak_low,x_lim_peak_high)
        plt.legend()
        plt.show()



    def integration(self, Peak_time_start_integration,Peak_time_end_integration,
                    TPR_time_start_integration, TPR_Temp_end_integration = 999.8,  Delta_n_H2_ref = 1.88e-6):

        self.Peak_time_start_integration = Peak_time_start_integration
        self.Peak_time_end_integration = Peak_time_end_integration
        self.TPR_time_start_integration = TPR_time_start_integration
        self.TPR_Temp_end_integration = TPR_Temp_end_integration
        self.Delta_n_H2_ref = Delta_n_H2_ref

        # integrate between the Ar peak and  the baseline 
        start_integration_index = np.where(self.data_array[:, 0] > Peak_time_start_integration)[0][0]
        end_integration_index = np.where(self.data_array[:, 0] > Peak_time_end_integration)[0][0]
        self.area_under_calib_peak = spi.trapezoid(y=self.data_array[start_integration_index:end_integration_index, 2] - self.baseline_peak(self.data_array[start_integration_index:end_integration_index, 0]), x=self.data_array[start_integration_index:end_integration_index, 0])
        # mol per area unit
        self.H2_cons_per_area = self.Delta_n_H2_ref / self.area_under_calib_peak
        print("Area under calib. peak : ", self.area_under_calib_peak, " a.u.*min")
        print("H2 consumption per area unit : ", self.H2_cons_per_area, " mol/(a.u.*min)")

        # integrate TPR 
        start_integration_index = np.where(self.data_array[:, 0] > TPR_time_start_integration)[0][0]
        end_integration_index = np.where(self.data_array[:, 1] > TPR_Temp_end_integration)[0][0]
        self.area_under_TPR_curve = spi.trapezoid(self.data_array[start_integration_index:end_integration_index, 2] - self.baseline_interp(self.data_array[start_integration_index:end_integration_index, 0]), self.data_array[start_integration_index:end_integration_index, 0])
        self.consumed_H2 = self.area_under_TPR_curve * self.H2_cons_per_area/ self.mass
        print("Consumed H2 : ", self.consumed_H2, " mol/gcat")
        print("Area under TPR curve : ", self.area_under_TPR_curve, " a.u.*min")


        