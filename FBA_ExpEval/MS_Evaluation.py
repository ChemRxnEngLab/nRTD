'''
This script calculates the molar fractions from the raw MS data.
It requires the following inputs:
- Calibration files for the MS
- Raw MS data
- GC points (optional) for a better calibration 
'''


import numpy as np
from tkinter import filedialog,messagebox
from tkinter.messagebox import askquestion
from tkinter.simpledialog import askstring,askinteger
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import ICIW_Plots
plt.style.use('ICIWstyle') 
import ICIW_Plots.colors as ICIWcolors
from numpy.typing import ArrayLike, NDArray
from datetime import datetime
import scipy.constants as const
import json




class MKS_MS_raw_to_mole_frac:
    def __init__(self, 
                CO2_or_CO_meth:str,         # CO or CO2 methanation ? enter: 'CO' / 'CO2,
                MS_exp_file_path:str,       # path to the MS data file
                MS_calib_file_paths:list,    # list of calibration file paths
                comp:NDArray,               # measured components via MS (can be checked via teh measured masses in the chosen MS program )
                
                # volumetric flowrates in Nml/min; use the same order as in the comp array (0 == time)
                V_dot_str_right:NDArray,    # Volumeflows of the right MFCs
                V_dot_str_left:NDArray,     # Volumeflows of the left MFCs
                V_dot_Matrix:NDArray,   # Volumeflows of the Matrix
                
                meas_Ar_at_40_and_20:bool = True , # is Ar measured at mass 40 and 20 ? - especially needed with Ne

                matrix_corr = 1.,      # Correction of MFC "H2 Matrix"
                counter = 0,

                # Ionisierungswahrscheinlichkeiten aus der Literatur - default values set for MKS MS
                He_ion = 0.14,
                Ar_ion = 1.2,    
                H2_ion = 0.44,
                CO_ion = 1.05,
                CO2_ion = 1.4,
                CH4_ion = 1.6,
                H2O_ion = 0.9642, # Von Dominik 
                C2H6_ion = 2.6,
                C3H8_ion = 3.7,
                Ne_ion = 0.23,


                ):

        self.CO2_or_CO_meth         = CO2_or_CO_meth
        self.MS_exp_file_path       = MS_exp_file_path
        self.MS_calib_file_paths    = MS_calib_file_paths
        self.comp                   = comp 
        self.V_dot_str_right        = V_dot_str_right
        self.V_dot_str_left         = V_dot_str_left
        self.V_dot_Matrix           = V_dot_Matrix
        self.matrix_corr            = matrix_corr
        self.counter                = counter
        self.He_ion                 = He_ion
        self.Ar_ion                 = Ar_ion
        self.H2_ion                 = H2_ion
        self.CO_ion                 = CO_ion
        self.CO2_ion                = CO2_ion
        self.CH4_ion                = CH4_ion
        self.H2O_ion                = H2O_ion
        self.C2H6_ion               = C2H6_ion
        self.C3H8_ion               = C3H8_ion
        self.Ne_ion                 = Ne_ion
        self.meas_Ar_at_40_and_20   = meas_Ar_at_40_and_20
        self.store_path             = os.path.dirname(os.path.dirname(self.MS_exp_file_path))

        self.p0                     = const.atm
        self.R                      = const.R
        self.T0                     = const.zero_Celsius
        
        'make each entry in species a variable and assign the corresponding index to it as value (e.g. Ar = 1, He = 2, H2 = 3, ...)'
        'give each species a specific color'
        self.color = []
        for cc in comp:
            exec(f"self." + f"{cc}  = {np.where(comp==cc)[0][0]} ")
            self.color.append(f"C" + f"{np.where(comp==cc)[0][0]}") 

        self.V_dot_str_right_total  = sum (self.V_dot_str_right)
        self.V_dot_str_left_total   = sum (self.V_dot_str_left)
        self.V_dot_Matrix_total     = sum (self.V_dot_Matrix)
        self.V_dot_in_analytics     = self.V_dot_str_right_total + self.V_dot_Matrix_total

        if self.V_dot_str_right_total != self.V_dot_str_right_total:
            raise ValueError('Volume flows right and left are not equal')
        
        
        
        
        
        
        self.cols_to_import = list(range(1,self.comp.size+1)) 

        
        # calcualtion of the RSF matrix with the MS calibration files
        for path_cal in self.MS_calib_file_paths:
            print('Used calibration file:',path_cal)
            self.ms_header_cal = np.loadtxt(path_cal,delimiter='\t',skiprows=2,usecols=self.cols_to_import,dtype=str,max_rows=1)
            self.ms_header_cal = np.char.replace(self.ms_header_cal, '"', '')
            self.ms_data_cal = np.loadtxt(path_cal,delimiter='\t',skiprows=3,usecols=self.cols_to_import)
            self.ms_data_avg_cal = np.mean(self.ms_data_cal,axis=0)
            ms_data_std_cal = np.std(self.ms_data_cal,axis=0)

            if len(self.ms_header_cal) != len(self.ms_data_avg_cal):
                raise ValueError('Header and data don´t fit!')
            
            ## Bestimmung der Konzentrationen aus Kalibrierdatei


            # template for titling the ms calibration files
            # 000_0Ar_000_0He_000_0H2_000_0H2M_000_0CO_000_0CO2_000_0CH4_000_0H2O_000_0C2_000_0C3_000_0Ne_otherdescription.txt
            # not included components can be omitted
            filename = os.path.basename(path_cal)
            Vdot = np.zeros_like(comp,dtype=float)
            print('\nVolume flows during calibration:')
            for i in range(len(comp)):
                ind_i = filename.find(comp[i]+'_') # search for the component i in the filename
                if ind_i != -1: # if ind_i equals -1, the component is not included in the file
                    if filename[ind_i-2] == '_': # one dezimal given
                        Vdot[i] = float(filename[ind_i-5:ind_i-2]) + float(filename[ind_i-1])/10 # read the volumeflows from the filename
                        print(comp[i],':',Vdot[i])
                        if i == np.where(comp=='H2')[0][0]: 
                            ind_matrix = filename.find('H2M_') # search H2 Matrix flow
                            H2_matrix = float(filename[ind_matrix-5:ind_matrix-2]) + float(filename[ind_matrix-1])/10
                            Vdot[i] = Vdot[i] + H2_matrix*self.matrix_corr
                            print('H2 Matrix:',H2_matrix,'*',self.matrix_corr,'=',H2_matrix*self.matrix_corr)
                            print('H2 total:',Vdot[i])
                    elif filename[ind_i-3] == '_': # two dezimals given
                        Vdot[i] = float(filename[ind_i-6:ind_i-3]) + float(filename[ind_i-2:ind_i])/100 # read the volumeflows from the filename
                        print(comp[i],':',Vdot[i])
                    

            x_i_cal = Vdot/sum(Vdot) # mol/mol

            # Ionisierungswahrscheinlichkeiten aus der Literatur
            He_ion = 0.14
            Ar_ion = 1.2    
            H2_ion = 0.44
            CO_ion = 1.05
            CO2_ion = 1.4
            CH4_ion = 1.6
            H2O_ion = 0.9642 # Von Dominik 
            C2H6_ion = 2.6
            C3H8_ion = 3.7
            Ne_ion = 0.23

            # Choose reference
            print('\nHe as reference')
            x_ref = x_i_cal[self.He] # x_ref = x_He as reference
            Idot_ref = self.ms_data_avg_cal[np.where(self.ms_header_cal=='Mass 4')[0][0]] # Idot_ref = Mass 4
            ion_ref = He_ion

            # Choose where to evaluate the RSF values
            eval_at = np.zeros((len(comp),len(self.ms_data_avg_cal))) # if eval_at is one at a position this combination is evaluated regarding the RSF(component,mass)
            if any(i == 'Ar' for i in comp): 
                print(meas_Ar_at_40_and_20)
                if self.meas_Ar_at_40_and_20:
                    print(20)
                    print(np.where(self.ms_header_cal=='Mass 20')[0][0])
                    eval_at[self.Ar, np.where(self.ms_header_cal=='Mass 20')[0][0]] = 1 # Ar,20
                eval_at[self.Ar,np.where(self.ms_header_cal=='Mass 40')[0][0]] = 1 # Ar,40
            if any(i == 'He' for i in comp):
                eval_at[self.He,np.where(self.ms_header_cal=='Mass 4')[0][0]] = 1 # He,4
            if any(i == 'H2' for i in comp):
                eval_at[self.H2,np.where(self.ms_header_cal=='Mass 2')[0][0]] = 1 # H2,2
            if any(i == 'CO' for i in comp):
                eval_at[self.CO,np.where(self.ms_header_cal=='Mass 28')[0][0]] = 1 # CO,28
                eval_at[self.CO,np.where(self.ms_header_cal=='Mass 12')[0][0]] = 1 # CO,28
            if any(i == 'CO2' for i in comp):
                eval_at[self.CO2,np.where(self.ms_header_cal=='Mass 44')[0][0]] = 1 # CO2,44
                eval_at[self.CO2,np.where(self.ms_header_cal=='Mass 28')[0][0]] = 1 # CO2,44
            if any(i == 'CH4' for i in comp):
                eval_at[self.CH4,np.where(self.ms_header_cal=='Mass 15')[0][0]] = 1 # CH4,15
                eval_at[self.CH4,np.where(self.ms_header_cal=='Mass 16')[0][0]] = 1 # CH4,16
            if any(i == 'H2O' for i in comp):
                eval_at[self.H2O,np.where(self.ms_header_cal=='Mass 16')[0][0]] = 1 # H2O,16
                eval_at[self.H2O,np.where(self.ms_header_cal=='Mass 18')[0][0]] = 1 # H2O,18
            if CO2_or_CO_meth == 'CO':
                if any(i == 'C2' for i in comp):
                    eval_at[self.C2,np.where(self.ms_header_cal=='Mass 26')[0][0]] = 1 # C2,26   # zuvor auf Masse 30 gemessen
                    eval_at[self.C2,np.where(self.ms_header_cal=='Mass 28')[0][0]] = 1 # C2,28
                if any(i == 'C3' for i in comp):
                    eval_at[self.C3,np.where(self.ms_header_cal=='Mass 28')[0][0]] = 1 # C3,28
                    eval_at[self.C3,np.where(self.ms_header_cal=='Mass 43')[0][0]] = 1 # C3,43
            if any(i == 'Ne' for i in comp):
                eval_at[np.where(comp=='Ne')[0][0],np.where(self.ms_header_cal=='Mass 20')[0][0]] = 1 # Ne,20

            # calculate RSF matrix
            RSF_temp = np.zeros_like(eval_at)
            for mm in range(eval_at.shape[1]):
                for i in range(len(x_i_cal)):
                    if x_i_cal[i] != 0: # only calculate RSF if component is present in the calibration measurement
                        RSF_temp[i,mm] = x_ref/x_i_cal[i]*self.ms_data_avg_cal[mm]/Idot_ref*eval_at[i,mm]

            if path_cal == self.MS_calib_file_paths[0]:
                RSF_all = np.array([RSF_temp])
            else:
                RSF_all = np.concatenate((RSF_all,np.array([RSF_temp])),axis=0)
            
            print()
            print('RSF',self.counter,'in %:\n',self.ms_header_cal,'\n',np.array(RSF_temp*100,dtype=int))
            self.counter += 1

        # Calculate final RSF matrix
        # Take the mean of values that occur more than one time
        RSF = np.true_divide(RSF_all.sum(0),(RSF_all!=0).sum(0))
        # RSF = np.nanmean(np.where(RSF_all!=0,RSF_all,np.nan),0)
        RSF = np.nan_to_num(RSF,nan=0.,posinf=0.,neginf=0.)

        print(self.ms_header_cal)

        # Use the literature values for all non calibrated components
        # Reference: Cracking Patterns
        # Methane   16/100 15/85 14/16 13/8 1/4
        # Ethane    28/100 27/33 30/26 26/23 29/21
        # Propane   29/100 28/59 27/38 44/26 43/22
        # Butane    43/100 29/44 27/37 28/32 41/27
        # CO        28/100 16/10 12/5 29/1
        non_calib = ~RSF.any(axis=1)
        print(non_calib)

        # Use the literature values for all non calibrated components
        # Reference: Cracking Patterns
        non_calib = ~RSF.any(axis=1)
        

        # non_calib = np.full((len(comp)),True) # uncomment to show the literatur data
        if any(i == 'Ar' for i in comp):  # Ar not calibrated
            if non_calib[self.Ar]:
                if self.meas_Ar_at_40_and_20:
                    RSF[self.Ar, np.where(self.ms_header_cal == 'Mass 20')[0][0]] = 0.1 * Ar_ion / ion_ref  # Ar,20
                RSF[self.Ar, np.where(self.ms_header_cal == 'Mass 40')[0][0]] = 1 * Ar_ion / ion_ref  # Ar,40
        if any(i == 'He' for i in comp):  # He not calibrated
            if non_calib[self.He]:
                RSF[self.He, np.where(self.ms_header_cal == 'Mass 4')[0][0]] = 1 * He_ion / ion_ref
        if any(i == 'H2' for i in comp):  # H2 not calibrated
            if non_calib[self.H2]:
                RSF[self.H2, np.where(self.ms_header_cal == 'Mass 2')[0][0]] = 1 * H2_ion / ion_ref
        if any(i == 'CO' for i in comp):  # CO not calibrated
            if non_calib[self.CO]:
                RSF[self.CO, np.where(self.ms_header_cal == 'Mass 28')[0][0]] = 1 * CO_ion / ion_ref
                RSF[self.CO, np.where(self.ms_header_cal == 'Mass 12')[0][0]] = 0.05 * CO_ion / ion_ref
        if any(i == 'CO2' for i in comp):  # CO2 not calibratsed
            if non_calib[self.CO2]:
                RSF[self.CO2, np.where(self.ms_header_cal == 'Mass 44')[0][0]] = 1 * CO2_ion / ion_ref  # CO2 at Mass 44
                RSF[self.CO2, np.where(self.ms_header_cal == 'Mass 28')[0][0]] = 0.11 * CO2_ion / ion_ref  # CO2 at Mass 28
        if any(i == 'CH4' for i in comp):  # CH4 not calibrated
            if non_calib[self.CH4]:
                RSF[self.CH4, np.where(self.ms_header_cal == 'Mass 15')[0][0]] = 0.85 * CH4_ion / ion_ref  # CH4 at Mass 15
                RSF[self.CH4, np.where(self.ms_header_cal == 'Mass 16')[0][0]] = 1 * CH4_ion / ion_ref  # CH4 at Mass 16
        if any(i == 'H2O' for i in comp):  # H2O not calibrated
            if non_calib[self.H2O]:
                RSF[self.H2O, np.where(self.ms_header_cal == 'Mass 16')[0][0]] = 0.01 * H2O_ion / ion_ref  # H2O at Mass 16
                RSF[self.H2O, np.where(self.ms_header_cal == 'Mass 18')[0][0]] = 1 * H2O_ion / ion_ref  # H2O at Mass 18
        if CO2_or_CO_meth == 'CO':
            if any(i == 'C2' for i in comp):  # C2 not calibrated
                if non_calib[self.C2]:
                    RSF[self.C2, np.where(self.ms_header_cal == 'Mass 26')[0][0]] = 0.24 * C2H6_ion / ion_ref  # C2 at Mass 26
                    RSF[self.C2, np.where(self.ms_header_cal == 'Mass 28')[0][0]] = 1 * C2H6_ion / ion_ref  # C2 at Mass 28
            if any(i == 'C3' for i in comp):  # C3 not calibrated
                if non_calib[self.C3]:
                    RSF[self.C3, np.where(self.ms_header_cal == 'Mass 28')[0][0]] = 0.59 * C3H8_ion / ion_ref  # C3 at Mass 28
                    RSF[self.C3, np.where(self.ms_header_cal == 'Mass 43')[0][0]] = 0.22 * C3H8_ion / ion_ref  # C3 at Mass 43
        if any(i == 'Ne' for i in comp):  # Ne not calibrated
            if non_calib[self.Ne]:
                RSF[self.Ne, np.where(self.ms_header_cal == 'Mass 20')[0][0]] = 1 * Ne_ion / ion_ref

        self.RSF = RSF
        print()
        print('RSF final in %:\n',np.array(self.RSF*100,dtype=int),'\n')
     

        # load the MS exp. data
        self.ms_exp      = pd.read_csv(self.MS_exp_file_path ,delimiter='\t',skiprows=2)
        self.pd_times         = pd.to_datetime(self.ms_exp['Time'],format='%d.%m.%Y %H:%M:%S.%f')
        ms_time_delta    = []
        for s in self.pd_times:
            ms_time_delta.append((s - self.pd_times[0]).total_seconds())
        self.ms_time_delta  = np.array((ms_time_delta))
        self.ms_exp.insert(1,'dt',ms_time_delta) 


    def calculate_each_comp(self):
        # Caclulate each component
        self.nt = np.zeros((len(self.comp),len(self.ms_time_delta)))
        # Argon
        if any(i == 'Ar' for i in self.comp):
            self.nt[self.Ar] = self.ms_exp['Mass 40']/self.RSF[self.Ar, np.where(self.ms_header_cal=='Mass 40')[0][0]]
        # Helium
        if any(i == 'He' for i in self.comp):
            self.nt[self.He] = self.ms_exp['Mass 4']/self.RSF[self.He, np.where(self.ms_header_cal=='Mass 4')[0][0]]
        # Hydrogen
        if any(i == 'H2' for i in self.comp):
            self.nt[self.H2] = self.ms_exp['Mass 2']/self.RSF[self.H2, np.where(self.ms_header_cal=='Mass 2')[0][0]]
        # Methane
        if any(i == 'CH4' for i in self.comp):
            self.nt[self.CH4] = self.ms_exp['Mass 15']/self.RSF[self.CH4, np.where(self.ms_header_cal=='Mass 15')[0][0]]
        # Water
        if any(i == 'H2O' for i in self.comp):
            self.nt[self.H2O] = self.ms_exp['Mass 18']/self.RSF[self.H2O, np.where(self.ms_header_cal=='Mass 18')[0][0]]

        # Carbon monoxide 
        if self.CO2_or_CO_meth == 'CO':
            # with C2 and C3 and CO2
        # C2 - exclude
            if any(i == 'C2' for i in self.comp):
                I_28_26_RSF = self.ms_exp['Mass 26'] * self.RSF[self.C2, np.where(self.ms_header_cal=='Mass 28')[0][0]] / self.RSF[self.C2, np.where(self.ms_header_cal=='Mass 26')[0][0]]
        # C3 - exclude
            if any(i == 'C3' for i in self.comp):
                I_28_43_RSF = self.ms_exp['Mass 43'] * self.RSF[self.C3, np.where(self.ms_header_cal=='Mass 28')[0][0]] / self.RSF[self.C3, np.where(self.ms_header_cal=='Mass 43')[0][0]]
        
        # CO2 - exclude
        if any(i == 'CO2' for i in self.comp):
            I_28_44_RSF =  self.ms_exp['Mass 44'] * self.RSF[self.CO2, np.where(self.ms_header_cal=='Mass 28')[0][0]] / self.RSF[self.CO2, np.where(self.ms_header_cal=='Mass 44')[0][0]]
            self.nt[self.CO2] = self.ms_exp['Mass 44']/self.RSF[self.CO2, np.where(self.ms_header_cal=='Mass 44')[0][0]]
            I_28_44_RSF =  self.nt[self.CO2] *0.11
            if self.CO2_or_CO_meth == 'CO':
                self.nt[self.CO]   = (self.ms_exp['Mass 28'] -  I_28_26_RSF  - I_28_43_RSF  - I_28_44_RSF ) / self.RSF[self.CO, np.where(self.ms_header_cal=='Mass 28')[0][0]]
            if self.CO2_or_CO_meth == 'CO2':
                self.nt[self.CO]   = (self.ms_exp['Mass 28'] - I_28_44_RSF ) / self.RSF[self.CO, np.where(self.ms_header_cal=='Mass 28')[0][0]]
        # Carbon dioxide
        

        
        # Ne
            # with Argon
        if any(i == 'Ne' for i in self.comp):
            if any(i == 'Ar' for i in self.comp):
                I_20_40_RSF = self.ms_exp['Mass 40'] * self.RSF[self.Ar, np.where(self.ms_header_cal=='Mass 20')[0][0]] / self.RSF[self.Ar, np.where(self.ms_header_cal=='Mass 40')[0][0]]
                self.nt[self.Ne] = (self.ms_exp['Mass 20'] - I_20_40_RSF) / self.RSF[self.Ne, np.where(self.ms_header_cal=='Mass 20')[0][0]]
            else:
                self.nt[self.Ne] = self.ms_exp['Mass 20']/self.RSF[self.Ne, np.where(self.ms_header_cal=='Mass 20')[0][0]]
        
        
           # nt_all = np.zeros((len(self.comp),len(nt_Ar)))
        
        
        
        # # C2
        # nt_C2 = np.zeros_like(nt_Ar)
        # # C3
        # nt_C3 = np.zeros_like(nt_Ar)
        # # C4
        # nt_C4 = np.zeros_like(nt_Ar)
        # # C5
        # nt_C5 = np.zeros_like(nt_Ar)
        # # C6
        # nt_C6 = np.zeros_like(nt_Ar)

        #nt_all = np.array([nt_Ar,nt_He,nt_H2,nt_CO,nt_CO2,nt_CH4,nt_H2O,nt_C2,nt_C3,nt_Ne,nt_C4,nt_C5,nt_C6])
        # Composition
        self.x_i = self.nt/np.sum(self.nt,axis=0)


    def H2O_stoich(self):
        '''
        calculates the H2O mole fraction based on the stoichiometry of the methanation reaction and WGS
        '''
        if self.CO2_or_CO_meth == 'CO2':
            return self.x_i[self.CH4, :] * 2 + self.x_i[self.CO, :] * 1
        if self.CO2_or_CO_meth == 'CO':
            return self.x_i[self.CH4, :] * 1 + self.x_i[self.C2, :] * 2 + self.x_i[self.C3, :] * 3 - self.x_i[self.CO2, :]



    def calibrate_with_GC(self, GC_file_path:str, gc_start_idx:int, gc_num_measurements:int,
                          set_x_lim_for_plot:bool = False, x_lim_low = 0, x_lim_high = 40000,
                          t_start_GC:float = 0, 
                            adj_RSF_CO2:float = 0,
                            adj_RSF_CO:float = 0,
                            adj_RSF_CH4:float = 0,
                            adj_RSF_H2O:float = 0,
                            adj_RSF_C2:float = 0,
                            adj_RSF_C3:float = 0,
                            adj_RSF_Ne:float = 0,
                            adj_RSF_Ar_20:float = 0, 
                            adj_RSF_Ar_40:float = 0,
                          ):
        '''
        Parameters:
        GC_file_path: str
            path to the GC data file
        gc_start_idx: int
            index of the GC data where the MS data to be be equal to the GC data starts
        gc_num_measurements: int
            number of GC measurements to be used for the calibration
        t_start_GC: float
            time to adjust GC data to MS data
        set_x_lim_for_plot: bool
            if True, the x-axis limits are set to x_lim_low and x_lim_high
        adj_RSF_XX
            if not equal to 0 this variable is used to adjust the RSF value for the component XX
        '''
        
        

        if set_x_lim_for_plot:
            x_lim_low = x_lim_low
            x_lim_high = x_lim_high
        else:
            x_lim_low = np.array((self.ms_exp['dt']))[0]
            x_lim_high = np.array((self.ms_exp['dt']))[-1]

        self.gc_start_idx = gc_start_idx
        self.gc_data = np.load(GC_file_path)
        self.gc_times = np.load(GC_file_path[:-12]+r'GC_times.npy')
        self.gc_num_measurements = gc_num_measurements
        self.gc_end_idx = gc_start_idx + self.gc_num_measurements
        self.t_start_GC = t_start_GC


        self.adj_RSF_CO2 = adj_RSF_CO2
        self.adj_RSF_CO = adj_RSF_CO
        self.adj_RSF_CH4 = adj_RSF_CH4
        self.adj_RSF_H2O = adj_RSF_H2O
        self.adj_RSF_C2 = adj_RSF_C2
        self.adj_RSF_C3 = adj_RSF_C3
        self.adj_RSF_Ne = adj_RSF_Ne
        self.adj_RSF_Ar_20 = adj_RSF_Ar_20
        self.adj_RSF_Ar_40 = adj_RSF_Ar_40

        
        
        
        if self.adj_RSF_CO2!=0:
            self.RSF[self.CO2, np.where(self.ms_header_cal=='Mass 44')[0][0]] = self.adj_RSF_CO2
        
        if self.adj_RSF_CO!=0:
            self.RSF[self.CO, np.where(self.ms_header_cal=='Mass 28')[0][0]] = self.adj_RSF_CO

        if self.adj_RSF_CH4!=0:
            self.RSF[self.CH4, np.where(self.ms_header_cal=='Mass 15')[0][0]] = self.adj_RSF_CH4

        if self.adj_RSF_C2!=0:
                self.RSF[self.C2, np.where(self.ms_header_cal=='Mass 26')[0][0]] = self.adj_RSF_C2

        if self.adj_RSF_C3!=0:
                self.RSF[self.C3, np.where(self.ms_header_cal=='Mass 43')[0][0]] = self.adj_RSF_C3

        if self.adj_RSF_H2O!=0:
            self.RSF[self.H2O, np.where(self.ms_header_cal=='Mass 18')[0][0]] = self.adj_RSF_H2O

        if self.adj_RSF_Ar_20!=0:
                    self.RSF[self.Ar, np.where(self.ms_header_cal=='Mass 20')[0][0]] = self.adj_RSF_Ar_20
        if self.adj_RSF_Ne!=0:
            self.RSF[self.Ne, np.where(self.ms_header_cal=='Mass 20')[0][0]] = self.adj_RSF_Ne
        
        if self.adj_RSF_Ar_40!=0:
            self.RSF[self.Ar, np.where(self.ms_header_cal=='Mass 40')[0][0]] = self.adj_RSF_Ar_40

        self.calculate_each_comp()

        
        
        if self.adj_RSF_CO2!=0 or self.adj_RSF_CO !=0 or self.adj_RSF_CH4!=0 or self.adj_RSF_H2O!=0 or self.adj_RSF_C2!=0 or self.adj_RSF_C3!=0 or self.adj_RSF_Ne!=0 or self.adj_RSF_Ar_20!=0 or self.adj_RSF_Ar_40!=0:
            print('RSF values are adjusted')


        
        plt.figure()
        plt.title('CO2')
        plt.plot(self.ms_exp['dt'], self.x_i[self.CO2, :], label='MS')
        plt.plot((self.gc_data[self.gc_start_idx:self.gc_end_idx, 0] - self.gc_data[self.gc_start_idx, 0]) + self.t_start_GC, self.gc_data[self.gc_start_idx:self.gc_end_idx, 5], label='GC')
        plt.legend(frameon=False)
        plt.xlim((x_lim_low, x_lim_high))
        #plt.ylim((0.04,0.053))
        plt.show()
        print('RSF_CO2: ',self.RSF[self.CO2, np.where(self.ms_header_cal=='Mass 44')[0][0]])
        
        
        plt.figure()
        plt.title('CO')
        plt.plot(self.ms_exp['dt'], self.x_i[self.CO, :], label='MS')
        plt.plot((self.gc_data[self.gc_start_idx:self.gc_end_idx, 0] - self.gc_data[self.gc_start_idx, 0]) + self.t_start_GC, self.gc_data[self.gc_start_idx:self.gc_end_idx, 4], label='GC')
        plt.legend(frameon=False)
        plt.xlim((x_lim_low, x_lim_high))
        if self.CO2_or_CO_meth == 'CO2':
            plt.ylim((0.00,0.0025))
        plt.show()
        print('RSF_CO: ', self.RSF[self.CO, np.where(self.ms_header_cal=='Mass 28')[0][0]])
        
        
        plt.figure()
        plt.title('CH4')
        plt.plot(self.ms_exp['dt'], self.x_i[self.CH4, :], label='MS')
        plt.plot((self.gc_data[self.gc_start_idx:self.gc_end_idx, 0] - self.gc_data[self.gc_start_idx, 0]) + self.t_start_GC, self.gc_data[self.gc_start_idx:self.gc_end_idx, 6], label='GC')
        plt.legend(frameon=False)
        plt.xlim((x_lim_low, x_lim_high))
        #plt.ylim((0.0055,0.008))
        plt.show()
        print('RSF_CH4: ', self.RSF[self.CH4, np.where(self.ms_header_cal=='Mass 15')[0][0]])



        if self.CO2_or_CO_meth == 'CO':

            
            plt.figure()
            plt.title('C2')
            plt.plot(self.ms_exp['dt'], self.x_i[self.C2, :], label='MS')
            plt.plot((self.gc_data[self.gc_start_idx:self.gc_end_idx, 0] - self.gc_data[self.gc_start_idx, 0]) + self.t_start_GC, self.gc_data[self.gc_start_idx:self.gc_end_idx, 8], label='GC')
            plt.legend(frameon=False)
            plt.xlim((x_lim_low, x_lim_high))
            #plt.ylim((0.00030,0.0004))
            plt.show()
            print('RSF_C2: ', self.RSF[self.C2, np.where(self.ms_header_cal=='Mass 26')[0][0]])

            #
            plt.figure()
            plt.title('C3')
            plt.plot(self.ms_exp['dt'], self.x_i[self.C3, :], label='MS')
            plt.plot((self.gc_data[self.gc_start_idx:self.gc_end_idx, 0] - self.gc_data[self.gc_start_idx, 0]) + self.t_start_GC, self.gc_data[self.gc_start_idx:self.gc_end_idx, 9], label='GC')
            plt.legend(frameon = False)
            plt.xlim((x_lim_low, x_lim_high))
            #plt.ylim((0.000125,0.0002))
            plt.show()
            print('RSF_C3: ', self.RSF[self.C3, np.where(self.ms_header_cal=='Mass 43')[0][0]])


        self.stoich_H2O = self.H2O_stoich()

        
        plt.figure()
        plt.title('H2O')
        plt.plot(self.ms_exp['dt'], self.x_i[self.H2O, :], label='MS')
        plt.plot(self.ms_exp['dt'], self.stoich_H2O, label='stoich')
        plt.legend(frameon=False)
        plt.xlim((x_lim_low, x_lim_high))
        #plt.ylim((0.006,0.009))
        plt.show()
        print('RSF_H2O: ', self.RSF[self.H2O, np.where(self.ms_header_cal=='Mass 18')[0][0]])
           
        
        self.ne_MFC = [self.V_dot_Matrix[self.Ne] / self.V_dot_in_analytics, self.V_dot_Matrix[self.Ne] / self.V_dot_in_analytics]
        plt.figure()
        plt.title('Ne')
        plt.plot(self.ms_exp['dt'], self.x_i[self.Ne, :])
        plt.plot([np.array((self.ms_exp['dt']))[0], np.array((self.ms_exp['dt']))[-5]], self.ne_MFC)
        plt.xlim((x_lim_low, x_lim_high))
        plt.ylim((0.005,0.018))
        #plt.xlim((50000,55000))
        plt.show()
        print('RSF_Ne: ', self.RSF[self.Ne, np.where(self.ms_header_cal=='Mass 20')[0][0]])
        print('RSF_Ar_20: ', self.RSF[self.Ar, np.where(self.ms_header_cal=='Mass 20')[0][0]])

        
        
        self.ar_MFC = [self.V_dot_str_right[self.Ar] / self.V_dot_in_analytics, self.V_dot_str_right[self.Ar] / self.V_dot_in_analytics]
        plt.figure()
        plt.title('Ar')
        plt.plot(self.ms_exp['dt'], self.x_i[self.Ar, :])
        plt.plot([np.array((self.ms_exp['dt']))[0], np.array((self.ms_exp['dt']))[-5]], self.ar_MFC)
        plt.xlim((x_lim_low, x_lim_high))
        plt.ylim((0.000,0.05))
        #plt.xlim((50000, 55000))
        plt.show()
        
        print('RSF_Ar_40: ', self.RSF[self.Ar, np.where(self.ms_header_cal=='Mass 40')[0][0]])
        
        

        self.ms_times_calib = np.array((self.ms_exp['dt']))
        self.ms_data_calib = self.x_i



    def calibrate_with_MFC_Ar_He(self,
                          set_x_lim_for_plot:bool = False, x_lim_low = 0, x_lim_high = 40000, 
                            adj_RSF_CO2:float = 0,
                            adj_RSF_CO:float = 0,
                            adj_RSF_CH4:float = 0,
                            adj_RSF_H2O:float = 0,
                            adj_RSF_C2:float = 0,
                            adj_RSF_C3:float = 0,
                            adj_RSF_Ne:float = 0,
                            adj_RSF_Ar_20:float = 0, 
                            adj_RSF_Ar_40:float = 0,
                            adj_RSF_He_4:float = 0
                          ):
        '''
        second calib via Ar and He MFCs (for RTD - experiments)

        Parameters:
        set_x_lim_for_plot: bool
            if True, the x-axis limits are set to x_lim_low and x_lim_high
        adj_RSF_XX
            if not equal to 0 this variable is used to adjust the RSF value for the component XX
        '''
        
        

        if set_x_lim_for_plot:
            x_lim_low = x_lim_low
            x_lim_high = x_lim_high
        else:
            x_lim_low = np.array((self.ms_exp['dt']))[0]
            x_lim_high = np.array((self.ms_exp['dt']))[-1]


        self.adj_RSF_CO2 = adj_RSF_CO2
        self.adj_RSF_CO = adj_RSF_CO
        self.adj_RSF_CH4 = adj_RSF_CH4
        self.adj_RSF_H2O = adj_RSF_H2O
        self.adj_RSF_C2 = adj_RSF_C2
        self.adj_RSF_C3 = adj_RSF_C3
        self.adj_RSF_Ne = adj_RSF_Ne
        self.adj_RSF_Ar_20 = adj_RSF_Ar_20
        self.adj_RSF_Ar_40 = adj_RSF_Ar_40
        self.adj_RSF_He_4  = adj_RSF_He_4

        # Adjust RSF values based on provided adjustments
       
        if self.adj_RSF_CO2!=0 or self.adj_RSF_CO !=0 or self.adj_RSF_CH4!=0 or self.adj_RSF_H2O!=0 or self.adj_RSF_C2!=0 or self.adj_RSF_C3!=0 or self.adj_RSF_Ne!=0 or self.adj_RSF_Ar_20!=0 or self.adj_RSF_Ar_40!=0:
            print('RSF values are adjusted')

        if self.adj_RSF_Ar_40!=0:
            self.RSF[self.Ar, np.where(self.ms_header_cal=='Mass 40')[0][0]] = self.adj_RSF_Ar_40
        if self.adj_RSF_He_4 != 0:
            self.RSF[self.He, np.where(self.ms_header_cal == 'Mass 4')[0][0]] = self.adj_RSF_He_4

        self.calculate_each_comp()

        
    
        self.ar_MFC = [self.V_dot_str_right[self.Ar] / self.V_dot_in_analytics, self.V_dot_str_right[self.Ar] / self.V_dot_in_analytics]
        plt.figure()
        plt.title('Ar')
        plt.plot(self.ms_exp['dt'], self.x_i[self.Ar, :])
        plt.plot([np.array((self.ms_exp['dt']))[0], np.array((self.ms_exp['dt']))[-1]], self.ar_MFC)
        plt.xlim((x_lim_low, x_lim_high))
        #plt.ylim((0.005,0.007))
        #plt.xlim((50000, 55000))
        plt.show()
        
        print('RSF_Ar_40: ', self.RSF[self.Ar, np.where(self.ms_header_cal=='Mass 40')[0][0]])
        

    
        self.he_MFC = [self.V_dot_str_left[self.He] / self.V_dot_in_analytics, self.V_dot_str_left[self.He] / self.V_dot_in_analytics]
        plt.figure()
        plt.title('He')
        plt.plot(self.ms_exp['dt'], self.x_i[self.He, :])
        plt.plot([np.array((self.ms_exp['dt']))[0], np.array((self.ms_exp['dt']))[-1]], self.he_MFC)
        plt.xlim((x_lim_low, x_lim_high))
        # plt.ylim((0.005, 0.007))
        # plt.xlim((50000, 55000))
        plt.show()

        print('RSF_He_4: ', self.RSF[self.He, np.where(self.ms_header_cal == 'Mass 4')[0][0]])

        
        

        self.ms_times_calib = np.array((self.ms_exp['dt']))
        self.ms_data_calib = self.x_i




    def show_prelim_results_after_all_calib(self):
        fig,ax = plt.subplots()
        for i in range(0,self.x_i.shape[0]):
            # ax.plot(np.arange(x_i.shape[1]),x_i[i,:]*sum(Vdot))
            #ax.plot(np.arange(x_i.shape[1]),x_i[i,:])
            ax.plot(self.ms_times_calib,self.ms_data_calib[i,:], color = self.color[i])
        ax.set(xlabel='time',ylabel='composition / mol/mol',title=os.path.basename(self.MS_exp_file_path)[:9])
        # ax.legend(ms_header)
        ax.legend(self.comp[1:])
        ax.set_title('Preliminary results!')
        ax.set_xlim()
        ax.set_ylim(-0.1,0.1)

    