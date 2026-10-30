'''
This script calculates the molar flow rates, the theoretical flow rates,  the transient molar flow rates from the mole fractions obtained in the skript MS_Evaluation.
It requires inputs:
- calibrated MS data (time and composition) see __init__
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
from scipy import interpolate
import json
from scipy.integrate import trapezoid
import scipy.stats as st
import pickle as pkl



class generate_PTK_cycle:
    def __init__(self, 
                CO2_or_CO_meth:str,         # CO or CO2 methanation ? enter: 'CO' / 'CO2,
                m_cat:float,                  # catalyst mass in g
                MS_times_with_GC:NDArray,       
                MS_composition_with_GC:NDArray,    
                comp:NDArray,               # measured components via MS (can be checked via teh measured masses in the chosen MS program )
                
                # volumetric flowrates in Nml/min; use the same order as in the comp array (0 == time)
                V_dot_str_right:NDArray,    # Volumeflows of the right MFCs
                V_dot_str_left:NDArray,     # Volumeflows of the left MFCs
                V_dot_Matrix:NDArray,   # Volumeflows of the Matrix
                t_H2_Phase:float,        # time of H2 Phase in s
                t_COx_Phase:float,          # time of COx Phase
                t_switch_Ar_sig:float,      # time between switching and Ar Signal in MS in s


                C = 0.95, # confidence interval
                set_x_ni_zero_H2_phase:bool = True # if a species is not fed, its molar flow rate is set to zero via linear scaling
                ):
   
        self.CO2_or_CO_meth         = CO2_or_CO_meth
        self.m_cat                  = m_cat
        self.comp                   = comp 
        self.times                  = MS_times_with_GC
        self.data                   = MS_composition_with_GC
        self.V_dot_str_right        = V_dot_str_right
        self.V_dot_str_left         = V_dot_str_left
        self.V_dot_Matrix           = V_dot_Matrix
        self.t_H2_Phase             = t_H2_Phase
        self.t_COx_Phase            = t_COx_Phase
        self.t_switch_Ar_sig        = t_switch_Ar_sig

        self.C                      = C

        self.set_x_ni_zero_H2_phase = set_x_ni_zero_H2_phase

        self.p0                     = const.atm
        self.R                      = const.R
        self.T0                     = const.zero_Celsius
        self.Nml_min_to_m3_s        = 1/6 * 1e-7 # conversion factor for Nml/min to m3/h

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

        
        self.t_per = self.t_H2_Phase + self.t_H2_Phase
        self.data 
        self.times 
        self.n_CO2_in = self.V_dot_str_right[self.CO2] * self.Nml_min_to_m3_s * self.p0 / self.R / self.T0
        self.n_CO_in  = self.V_dot_str_right[self.CO] * self.Nml_min_to_m3_s * self.p0 / self.R / self.T0

    def scale_zero_to_max(self,y_array:NDArray) -> NDArray:
    
        y_min_i = np.min(y_array)
        y_max_i = np.max(y_array)
        # linear function to set min to 0 and does not change the max value
        # y = a*x + b
        #a = y_max_i/(y_max_i - y_min_i)
        #b = - y_min_i * a
        #return a * y_array + b
        scaled = (y_array - y_min_i) / (y_max_i - y_min_i) * y_max_i
        return scaled
    
    

    def find_end_for_averaging(self, x_lim_low = 8000 ,x_lim_high = 20500): 
        'end time for averaging is needed for the next function'
        print('Check plot to see the end of the PTK experiment - this value is end_for_averaging for function: calculate_x_avg_x_std_x_CI')
        plt.figure()
        plt.plot(self.times,self.data[self.Ar])
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel(r'$x_\mathrm{Ar}\,/\,\mathrm{1}$')
        plt.xlim(x_lim_low,x_lim_high)
        plt.show()
    

    def calculate_x_avg_x_std_x_CI(self, end_time_avg, start_H2_Phase=True, num_avrg_cycles = 5):
        ''''
        end_time_avg : time, when averaging should end - start time is calculated with the period length
        std : Standard Deviation
        CI : Confidence Intervall 
        start_H2_Phase : if False - Phases are changed
        '''

        self.num_avrg_cycles = num_avrg_cycles
        self.end_time_avg   = end_time_avg
        self.start_time_avg = end_time_avg - (self.num_avrg_cycles + 1)*self.t_per
        self.start_H2_Phase = start_H2_Phase

        idx_start           = np.where(self.times > self.start_time_avg)[0][0] # next(k for k, value in enumerate(times[0,:]) if value > start_time_avg) # Index start average
        idx_end             = np.where(self.times > end_time_avg)[0][0] # next(k for k, value in enumerate(times[0,:]) if value > end_time_avg) # Index end average


        # logical array 

        # save relevant data and adjust the time
        data_relevant       = self.data[:,idx_start:idx_end]
        times_relevant      = self.times[idx_start:idx_end] - self.times[idx_start] # set average start time to 0
        n_period            = times_relevant/self.t_per  # time normalized on the period time --> counter for each full period



        #n_last_period       = int(n_period[-1]) # number of cycles in relevant data
        self.num_cycles          = int(float(n_period[-1])) # number of cycles in relevant data
        self.idx_split           = np.empty(self.num_cycles)
        self.start_period        = np.empty(self.num_cycles)
        #start_time = np.empty(n_last_period)
        #start_time_min1 = np.empty(n_last_period)


        for i in range(self.num_cycles):
            self.idx_split[i]    = next(k for k, value in enumerate(n_period) if value>(i+1)) # counts how many measurement points are in each cycle - uses normalized time 
            self.start_period[i] = n_period[int(self.idx_split[i])] # the periods start at a slightly different time as the time step is not 100% constant, just to see how much difference

        #print(times_relevant)

        # interpolation, to get equally distributed values that can be averaged more easily ### as functions
        # comp = np.array(['Ar','He','H2','CO','CO2','CH4','H2O','C2H6','C3H8','Ne','C4','C5','C6','C2en','C3en']) # Component names
        self.x_Ar = interpolate.make_interp_spline(times_relevant,data_relevant[self.Ar], k = 1)
        self.x_He = interpolate.make_interp_spline(times_relevant,data_relevant[self.He], k = 1)
        self.x_H2 = interpolate.make_interp_spline(times_relevant,data_relevant[self.H2], k = 1)
        self.x_CO = interpolate.make_interp_spline(times_relevant,data_relevant[self.CO], k = 1)
        self.x_CO2 = interpolate.make_interp_spline(times_relevant,data_relevant[self.CO2], k = 1)
        self.x_CH4 = interpolate.make_interp_spline(times_relevant,data_relevant[self.CH4], k = 1)
        self.x_H2O = interpolate.make_interp_spline(times_relevant,data_relevant[self.H2O], k = 1)
        self.x_C2H6 = interpolate.make_interp_spline(times_relevant,data_relevant[self.C2], k = 1)
        self.x_C3H8 = interpolate.make_interp_spline(times_relevant,data_relevant[self.C3], k = 1)
        self.x_Ne = interpolate.make_interp_spline(times_relevant,data_relevant[self.Ne], k = 1)
        #self.x_C4 = interpolate.make_interp_spline(data_relevant[:,0],data_relevant[:,C4], k = 1)
        #self.x_C5 = interpolate.make_interp_spline(data_relevant[:,0],data_relevant[:,C5], k = 1)
        #self.x_C2en = interpolate.make_interp_spline(data_relevant[:,0],data_relevant[:,C2en], k = 1)
        #self.x_C3en = interpolate.make_interp_spline(data_relevant[:,0],data_relevant[:,C3en], k = 1)


        print('Number of averaged limit cycles',self.num_cycles)
        self.n_points = int(self.idx_split[0])
        print('Number of data points in one period:',self.n_points)

        # 95 % of all values are within the confidence interval
        self.tails           = 2 
        self.alpha           = 1 - self.C # significance level
        self.q               = 1 - (self.alpha / self.tails) # quantile, cumulative probability
        self.dof             = self.num_cycles - 1 # degrees of freedom
        self.t_star          = st.t.ppf(self.q,self.dof) # quantile function of the t distribution --- ppf = Percent point function 

        # time array
        self.t_lin           = np.linspace(0,self.t_per,self.n_points)


        self.x_Ar_split      = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_Ar_avg        = np.empty(self.n_points)
        self.x_Ar_std        = np.empty(self.n_points)
        self.x_Ar_CI         = np.empty(self.n_points)
        # self.x_Ar_CI_up = np.empty(self.n_points)
        # self.x_Ar_CI_low = np.empty(self.n_points)

        self.x_He_split = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_He_avg = np.empty(self.n_points)
        self.x_He_std = np.empty(self.n_points)
        self.x_He_CI = np.empty(self.n_points)
        # self.x_He_CI_up = np.empty(self.n_points)
        # self.x_He_CI_low = np.empty(self.n_points)

        self.x_H2_split = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_H2_avg = np.empty(self.n_points)
        self.x_H2_std = np.empty(self.n_points)
        self.x_H2_CI = np.empty(self.n_points)
        # self.x_H2_CI_up = np.empty(self.n_points)
        # self.x_H2_CI_low = np.empty(self.n_points)

        self.x_CO_split = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_CO_avg = np.empty(self.n_points)
        self.x_CO_std = np.empty(self.n_points)
        self.x_CO_CI = np.empty(self.n_points)
        # self.x_CO_CI_up = np.empty(self.n_points)
        # self.x_CO_CI_low = np.empty(self.n_points)

        self.x_CO2_split = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_CO2_avg = np.empty(self.n_points)
        self.x_CO2_std = np.empty(self.n_points)
        self.x_CO2_CI = np.empty(self.n_points)
        # self.x_CO2_CI_up = np.empty(self.n_points)
        # self.x_CO2_CI_low = np.empty(self.n_points)

        self.x_CH4_split = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_CH4_avg = np.empty(self.n_points)
        self.x_CH4_std = np.empty(self.n_points)
        self.x_CH4_CI = np.empty(self.n_points)
        # self.x_CH4_CI_up = np.empty(self.n_points)
        # self.x_CH4_CI_low = np.empty(self.n_points)

        self.x_H2O_split = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_H2O_avg = np.empty(self.n_points)
        self.x_H2O_std = np.empty(self.n_points)
        self.x_H2O_CI = np.empty(self.n_points)
        # self.x_H2O_CI_up = np.empty(self.n_points)
        # self.x_H2O_CI_low = np.empty(self.n_points)

        self.x_C2H6_split = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_C2H6_avg = np.empty(self.n_points)
        self.x_C2H6_std = np.empty(self.n_points)
        self.x_C2H6_CI = np.empty(self.n_points)
        # self.x_C2H6_CI_up = np.empty(self.n_points)
        # self.x_C2H6_CI_low = np.empty(self.n_points)

        self.x_C3H8_split = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_C3H8_avg = np.empty(self.n_points)
        self.x_C3H8_std = np.empty(self.n_points)
        self.x_C3H8_CI = np.empty(self.n_points)
        # self.x_C3H8_CI_up = np.empty(self.n_points)
        # self.x_C3H8_CI_low = np.empty(self.n_points)

        self.x_Ne_split = np.empty((self.num_cycles,len(self.t_lin)))
        self.x_Ne_avg = np.empty(self.n_points)
        self.x_Ne_std = np.empty(self.n_points)
        self.x_Ne_CI = np.empty(self.n_points)
        # self.x_Ne_CI_up = np.empty(self.n_points)
        # self.x_Ne_CI_low = np.empty(self.n_points)

        ### calling the interpolation functions for the specified time intervals  
        for j in range(self.num_cycles):         # cycles 
            for i  in range(self.n_points):      # data points 
                self.x_Ar_split[j,i]     = self.x_Ar(self.t_lin[i]+j*self.t_per)
                self.x_He_split[j,i]     = self.x_He(self.t_lin[i]+j*self.t_per)
                self.x_H2_split[j,i]     = self.x_H2(self.t_lin[i]+j*self.t_per)
                self.x_CO_split[j,i]     = self.x_CO(self.t_lin[i]+j*self.t_per)
                self.x_CO2_split[j,i]    = self.x_CO2(self.t_lin[i]+j*self.t_per)
                self.x_CH4_split[j,i]    = self.x_CH4(self.t_lin[i]+j*self.t_per)
                self.x_H2O_split[j,i]    = self.x_H2O(self.t_lin[i]+j*self.t_per)
                self.x_C2H6_split[j,i]   = self.x_C2H6(self.t_lin[i]+j*self.t_per)
                self.x_C3H8_split[j,i]   = self.x_C3H8(self.t_lin[i]+j*self.t_per)
                self.x_Ne_split[j,i]     = self.x_Ne(self.t_lin[i]+j*self.t_per)
            
        # savename_split = input('How should the save file be named? ')
        # filepath_split = r'C:\Users\MaxGäßler\Documents\GitHub\sh-10-22\Experimentell\saved_data\ ' + savename_split
        # savedata_split = [self.x_Ar_split,self.x_He_split,self.x_H2_split,self.x_CO_split,self.x_CO2_split,self.x_CH4_split,self.x_H2O_split,self.x_C2H6_split,self.x_C3H8_split,self.x_Ne_split,]
        # np.save(filepath_split,savedata_split)

        for i in range(self.n_points):
            # arithmetic mean
            self.x_Ar_avg[i] = np.mean(self.x_Ar_split[:,i])
            self.x_He_avg[i] = np.mean(self.x_He_split[:,i])
            self.x_H2_avg[i] = np.mean(self.x_H2_split[:,i])
            self.x_CO_avg[i] = np.mean(self.x_CO_split[:,i])
            self.x_CO2_avg[i] = np.mean(self.x_CO2_split[:,i])
            self.x_CH4_avg[i] = np.mean(self.x_CH4_split[:,i])
            self.x_H2O_avg[i] = np.mean(self.x_H2O_split[:,i])
            self.x_C2H6_avg[i] = np.mean(self.x_C2H6_split[:,i])
            self.x_C3H8_avg[i] = np.mean(self.x_C3H8_split[:,i])
            self.x_Ne_avg[i] = np.mean(self.x_Ne_split[:,i])
            
            # standard deviation
            self.x_Ar_std[i] = np.std(self.x_Ar_split[:,i], ddof=1)
            self.x_He_std[i] = np.std(self.x_He_split[:,i], ddof=1)
            self.x_H2_std[i] = np.std(self.x_H2_split[:,i], ddof=1)
            self.x_CO_std[i] = np.std(self.x_CO_split[:,i], ddof=1)
            self.x_CO2_std[i] = np.std(self.x_CO2_split[:,i], ddof=1)
            self.x_CH4_std[i] = np.std(self.x_CH4_split[:,i], ddof=1)
            self.x_H2O_std[i] = np.std(self.x_H2O_split[:,i], ddof=1)
            self.x_C2H6_std[i] = np.std(self.x_C2H6_split[:,i], ddof=1)
            self.x_C3H8_std[i] = np.std(self.x_C3H8_split[:,i], ddof=1)
            self.x_Ne_std[i] = np.std(self.x_Ne_split[:,i], ddof=1)
            
            # confidence interval
            # margin of error = t*std/sqrt(N)
            self.x_Ar_CI[i] = self.t_star * self.x_Ar_std[i] / np.sqrt(self.num_cycles)
            self.x_He_CI[i] = self.t_star * self.x_He_std[i] / np.sqrt(self.num_cycles)
            self.x_H2_CI[i] = self.t_star * self.x_H2_std[i] / np.sqrt(self.num_cycles)
            self.x_CO_CI[i] = self.t_star * self.x_CO_std[i] / np.sqrt(self.num_cycles)
            self.x_CO2_CI[i] = self.t_star * self.x_CO2_std[i] / np.sqrt(self.num_cycles)
            self.x_CH4_CI[i] = self.t_star * self.x_CH4_std[i] / np.sqrt(self.num_cycles)
            self.x_H2O_CI[i] = self.t_star * self.x_H2O_std[i] / np.sqrt(self.num_cycles)
            self.x_C2H6_CI[i] = self.t_star * self.x_C2H6_std[i] / np.sqrt(self.num_cycles)
            self.x_C3H8_CI[i] = self.t_star * self.x_C3H8_std[i] / np.sqrt(self.num_cycles)
            self.x_Ne_CI[i] = self.t_star * self.x_Ne_std[i] / np.sqrt(self.num_cycles)
            
        
 

        self.x_avg = np.vstack((self.x_Ar_avg,self.x_He_avg,self.x_H2_avg,self.x_CO_avg,self.x_CO2_avg,self.x_CH4_avg,
                self.x_H2O_avg,self.x_C2H6_avg,self.x_C3H8_avg,self.x_Ne_avg))
        self.x_std = np.vstack((self.x_Ar_std,self.x_He_std,self.x_H2_std,self.x_CO_std,self.x_CO2_std,self.x_CH4_std,
                self.x_H2O_std,self.x_C2H6_std,self.x_C3H8_std,self.x_Ne_std))
        self.x_CI = np.vstack((self.x_Ar_CI,self.x_He_CI,self.x_H2_CI,self.x_CO_CI,self.x_CO2_CI,self.x_CH4_CI,
                self.x_H2O_CI,self.x_C2H6_CI,self.x_C3H8_CI,self.x_Ne_CI))
        

  
        # plt.figure()
        # for i in range(len(self.x_avg[:,0])):
        #     plt.plot(self.t_lin,self.x_avg[i,:],label=self.comp[i], color = self.color[i])
        # #plt.ylim(-0.00,0.01)
        # plt.legend()
        # plt.xlabel(r'$t\,/\,\mathrm{s}$')
        # plt.ylabel(r'$x_i\,/\,1$')
        # #plt.savefig('Test1.svg')
        # plt.show()

        # plt.figure()
        # for i in range(len(self.x_avg[:,0])):
        #     plt.plot(self.t_lin,self.x_avg[i,:],label=self.comp[i],color = self.color[i])
        # plt.ylim(-0.00,0.0005)
        # plt.xlabel(r'$t\,/\,\mathrm{s}$')
        # plt.ylabel(r'$x_i\,/\,1$')
        # plt.legend()
        # #plt.savefig('Test2.png')
        # plt.show()

        # plt.figure()
        # plt.plot(self.t_lin,self.x_avg[self.Ar,:],label=self.comp[self.Ar], color = self.color[self.Ar])
        # plt.ylim(-0.00,0.0005)
        # plt.xlabel(r'$t\,/\,\mathrm{s}$')
        # plt.ylabel(r'$x_\mathrm{Ar}\,/\,1$')
        # plt.legend()
        # #plt.savefig('Test2.png')
        # plt.show()


        '''
        orders the values, that they start with the COx Phase
        time_delay: time between switching and Ar Signal 
        '''
        
        self.t_lin
        
        self.num_values_offset = np.where(self.t_lin > self.t_switch_Ar_sig)[0][0]  # number of measurements during the delay between switching and Ar Signal

        if self.x_avg[self.Ar,0] - np.min(self.x_avg[self.Ar] )> 1e-4:
            # change start to hydrogen phase
            cut = np.where(self.x_avg[self.Ar]- np.min(self.x_avg[self.Ar,0] ) < 1e-4)[0][0]
            self.x_avg = np.hstack((self.x_avg[:,cut:],self.x_avg[:,:cut]))
            self.x_std = np.hstack((self.x_std[:,cut:],self.x_std[:,:cut]))
            self.x_CI = np.hstack((self.x_CI[:,cut:],self.x_CI[:,:cut]))

        # begin with H2- Phase
        # cutting due to threshold of Ar 
        t_cut_index = np.where(self.x_avg[self.Ar]- np.min(self.x_avg[self.Ar,0] ) > 1e-4)[0][0]
        # identify if cuts needs to be at the beginning or end of interval
        self.length_H2_Phase = int(self.x_avg[0].size * (self.t_H2_Phase/self.t_per))
        if start_H2_Phase:
            if t_cut_index > self.num_values_offset:
                cut_ind = int(t_cut_index - self.num_values_offset)
            else:
                cut_ind = int(self.x_avg[0].size-self.num_values_offset+t_cut_index)
        else:
            if t_cut_index > self.num_values_offset + self.length_H2_Phase:
                cut_ind = int(t_cut_index - self.num_values_offset - self.length_H2_Phase)
            else:
                cut_ind = int(self.x_avg[0].size - self.num_values_offset + t_cut_index - self.length_H2_Phase)



        print(cut_ind)
        print(t_cut_index)

        # cutting and stacking
        # comp # time
        self.x_avg = np.hstack((self.x_avg[:,cut_ind:],self.x_avg[:,:cut_ind]))
        self.x_std = np.hstack((self.x_std[:,cut_ind:],self.x_std[:,:cut_ind]))
        self.x_CI = np.hstack((self.x_CI[:,cut_ind:],self.x_CI[:,:cut_ind]))

        
    def scale_base_to_qss1(self,y_array:NDArray, qss1_value:float, qss2_value:float) -> NDArray:
        
        # linear function to set the base_value to zero and not change the max value - as base value the qss in the H2-Phase is used for CO,CO2,CH4,H2O
        #1 subtract base value from y array
        y_array = y_array - qss2_value
        # get the new qss1 value of CO2 Phase
        qss1_value_new = np.mean(y_array[(self.idx_qss-10):(self.idx_qss)])
        #2 apply scaling: the qss_1 of the y_array should now be scaled to y_max_i 
        y_array = y_array/  qss1_value_new * qss1_value
        return y_array

  


    def cal_qss_norm_n_trans(self, t_qss, balance_with_Ne = False ):#'True'):
        '''
        calculates the quasi steady states (qss)
        t_qss: time in s, when qss is reached --- (averaging for qss from t_qss-10(values) to t_qss)
                calculated via t_COx_Phase 
        '''
        self.t_qss      = t_qss

        self.idx_qss    = np.where(self.t_lin > t_qss)[0][0] 

        # COx Phase
        self.x_qss      = np.empty(len(self.x_avg[:,0]))
        self.x_qss[0]   = np.mean(self.x_avg[0,(self.idx_qss-10):(self.idx_qss)]) 
        for i in range(len(self.x_avg[:,0])):
            self.x_qss[i] = np.mean(self.x_avg[i,(self.idx_qss-10):self.idx_qss])
        # H2 Phase
        self.x_qss2     = np.empty(len(self.x_avg[:,0]))
        self.x_qss2[0]   = np.mean(self.x_avg[0,-20:-10])
        for i in range(len(self.x_avg[:,0])):
            self.x_qss2[i] = np.mean(self.x_avg[i,-20:-10])
        
        # Adjust baseline for CO, CO2, CH4, H2O based on qss in H2-Phase
        self.x_avg[self.CO,:] = self.scale_base_to_qss1(self.x_avg[self.CO,:], self.x_qss[self.CO], self.x_qss2[self.CO])  # CO
        self.x_avg[self.CO2,:] = self.scale_base_to_qss1(self.x_avg[self.CO2,:], self.x_qss[self.CO2], self.x_qss2[self.CO2])  # CO2
        self.x_avg[self.CH4,:] = self.scale_base_to_qss1(self.x_avg[self.CH4,:], self.x_qss[self.CH4], self.x_qss2[self.CH4])  # CH4
        self.x_avg[self.H2O,:] = self.scale_base_to_qss1(self.x_avg[self.H2O,:], self.x_qss[self.H2O], self.x_qss2[self.H2O])  # H2O

        # plt.figure()
        # plt.plot(self.t_lin,self.x_avg[self.CO,:],label=self.comp[self.CO], color = self.color[self.CO])
        # plt.plot(self.t_lin,self.x_avg[self.CO2,:],label=self.comp[self.CO2], color = self.color[self.CO2])
        # plt.plot(self.t_lin,self.x_avg[self.CH4,:],label=self.comp[self.CH4], color = self.color[self.CH4])
        # plt.plot(self.t_lin,self.x_avg[self.H2O,:],label=self.comp[self.H2O], color = self.color[self.H2O])
        # plt.ylim(-0.00,0.01)    
        # plt.legend(loc = 'upper right')


        # normalization on the qss values 

        self.x_avg_norm = np.empty_like(self.x_avg)
        self.x_min = np.empty_like(self.x_qss)



        # for i in range(len(self.x_avg[:,0])):
        #     self.x_min[i] = min(self.x_avg[i,:])
        #     self.x_avg_norm[i,:] = (self.x_avg[i,:]-self.x_min[i]) / (self.x_qss[i]-self.x_min[i])
        
        # plt.figure()
        # for i in range(len(self.x_avg[:,0])):
        #     plt.plot(self.t_lin,self.x_avg[i,:],label=self.comp[i], color = self.color[i])
        #     plt.legend(loc = 'upper right')
        #     #plt.savefig('Test2.png')
        # plt.xlabel(r'$t\,/\,\mathrm{s}$')
        # plt.ylabel(r'$x_i\,/\,1$')
        # plt.show()

        # plt.figure()
        # for i in range(len(self.x_avg[:,0])):
        #     plt.plot(self.t_lin,self.x_avg[i,:],label=self.comp[i], color = self.color[i])
        #     plt.ylim(-0.00,0.0005)
        #     plt.legend(loc = 'upper right')
        #     #plt.savefig('Test2.png')
        # plt.xlabel(r'$t\,/\,\mathrm{s}$')
        # plt.ylabel(r'$x_i\,/\,1$')
        # plt.show()

        plt.figure()
        plt.plot(self.t_lin,self.x_avg[self.Ar,:],label=self.comp[self.Ar], color = self.color[self.Ar])
        #plt.ylim(-0.00,0.0005)
        plt.legend(loc = 'upper right')
        #plt.savefig('Test2.png')
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel(r'$x_\mathrm{Ar}\,/\,1$')
        plt.show()



        '''
        calculation of the measured molar flowrates with Ne 
        '''

        if balance_with_Ne:
            self.V_out = self.V_dot_Matrix[self.Ne] / self.x_avg[self.Ne]
            self.n_out_ges = self.V_out * self.Nml_min_to_m3_s * self.p0 / self.R / self.T0 # n = p0V/RT0, 
            self.n_avg = self.n_out_ges * self.x_avg  
        else:
            self.n_out_ges = self.V_dot_in_analytics * self.Nml_min_to_m3_s * self.p0 / self.R / self.T0
            self.n_avg = self.n_out_ges * self.x_avg
        

        


        '''
        calculates the transient molar flow rates
        '''
        self.n_dot_out_R        = self.n_avg[0,:] + self.n_avg[1,:] +self.n_avg[2,:]+ self.n_avg[3,:] + self.n_avg[4,:] + self.n_avg[5,:] + self.n_avg[6,:] + self.n_avg[9,:]- (self.V_dot_Matrix_total/self.Nml_min_to_m3_s*self.p0)/(self.R*self.T0)  # n_i is already averaged -> 
        self.n_dot_in_R         = (self.V_dot_str_right_total /6*1e-7*self.p0)/(self.R*self.T0)
        
        # the averaged limit cycle is cut into COx_Phase and H2_Phase
        self.idx_COx            = np.where(self.t_lin > self.t_COx_Phase)[0][0]


        # if self.CO2_or_CO_meth == 'CO2' and self.set_x_ni_zero_H2_phase:
        #     self.n_avg[self.CO] = self.scale_zero_to_max(self.n_avg[self.CO,:])
        #     self.n_avg[self.CH4] = self.scale_zero_to_max(self.n_avg[self.CH4,:])
        #     self.n_avg[self.H2O] = self.scale_zero_to_max(self.n_avg[self.H2O,:])
        # elif self.CO2_or_CO_meth == 'CO' and self.set_x_ni_zero_H2_phase:
        #     self.n_avg[self.CO2] = self.scale_zero_to_max(self.n_avg[self.CO2,:])
        #     self.n_avg[self.CH4] = self.scale_zero_to_max(self.n_avg[self.CH4,:])
        #     self.n_avg[self.H2O] = self.scale_zero_to_max(self.n_avg[self.H2O,:])

        self.ni_COx_Phase       = self.n_avg[:,:self.idx_COx]
        self.ni_H2_Phase        = self.n_avg[:,self.idx_COx:]

        # calculation of transient molar flow rates according to  Meyer et al. 2021 (10.1016/j.cherd.2021.07.011)
        self.ni_qss_COx_Phase           = np.average(self.ni_COx_Phase [:,-10:-1], axis = 1) # Quasi-steady state
        self.ni_qss_H2_Phase            = np.average(self.ni_H2_Phase[:,-10:-1], axis = 1) # Quasi-steady state
    




        self.ni_out                     = np.hstack((self.ni_COx_Phase ,self.ni_H2_Phase))
        self.F_i                        = (self.ni_out.T -  self.ni_qss_COx_Phase)/ (self.ni_qss_H2_Phase - self.ni_qss_COx_Phase)
        self.ni_RTD                     = np.multiply(self.F_i[:,0], ((self.ni_qss_H2_Phase - self.ni_qss_COx_Phase).reshape(10,1))) + self.ni_qss_COx_Phase.reshape(10,1)
        self.ni_trans                   = self.ni_out - self.ni_RTD
        self.ni_norm                    = ((self.ni_out.T - self.ni_qss_H2_Phase)/(self.ni_qss_COx_Phase-self.ni_qss_H2_Phase)).T


        # if self.CO2_or_CO_meth == 'CO2' and self.set_x_ni_zero_H2_phase:
            
        #     self.ni_RTD[self.CO] = self.scale_zero_to_max(self.ni_RTD[self.CO])
        #     self.ni_RTD[self.CH4] = self.scale_zero_to_max(self.ni_RTD[self.CH4])
        #     self.ni_RTD[self.H2O] = self.scale_zero_to_max(self.ni_RTD[self.H2O])

        #     self.ni_out[self.CO] = self.scale_zero_to_max(self.ni_out[self.CO])
        #     self.ni_out[self.CH4] = self.scale_zero_to_max(self.ni_out[self.CH4])
        #     self.ni_out[self.H2O] = self.scale_zero_to_max(self.ni_out[self.H2O])

        #     self.x_CI[self.CO] = self.scale_zero_to_max(self.x_CI[self.CO,:])
        #     self.x_CI[self.CH4] = self.scale_zero_to_max(self.x_CI[self.CH4,:])
        #     self.x_CI[self.H2O] = self.scale_zero_to_max(self.x_CI[self.H2O,:])

        #     self.n_avg[self.CO] = self.scale_zero_to_max(self.n_avg[self.CO,:])
        #     self.n_avg[self.CH4] = self.scale_zero_to_max(self.n_avg[self.CH4,:])
        #     self.n_avg[self.H2O] = self.scale_zero_to_max(self.n_avg[self.H2O,:])

        #     self.ni_norm[self.CO] = self.scale_zero_to_max(self.ni_norm[self.CO])
        #     self.ni_norm[self.CH4] = self.scale_zero_to_max(self.ni_norm[self.CH4])
        #     self.ni_norm[self.H2O] = self.scale_zero_to_max(self.ni_norm[self.H2O])

        # elif self.CO2_or_CO_meth == 'CO' and self.set_x_ni_zero_H2_phase:
        #     self.ni_RTD[self.CO2] = self.scale_zero_to_max(self.ni_RTD[self.CO2])
        #     self.ni_RTD[self.CH4] = self.scale_zero_to_max(self.ni_RTD[self.CH4])
        #     self.ni_RTD[self.H2O] = self.scale_zero_to_max(self.ni_RTD[self.H2O])

        #     self.ni_out[self.CO2] = self.scale_zero_to_max(self.ni_out[self.CO2])
        #     self.ni_out[self.CH4] = self.scale_zero_to_max(self.ni_out[self.CH4])
        #     self.ni_out[self.H2O] = self.scale_zero_to_max(self.ni_out[self.H2O])

        #     self.n_avg[self.CO2] = self.scale_zero_to_max(self.n_avg[self.CO2,:])
        #     self.n_avg[self.CH4] = self.scale_zero_to_max(self.n_avg[self.CH4,:])
        #     self.n_avg[self.H2O] = self.scale_zero_to_max(self.n_avg[self.H2O,:])

        #     self.ni_norm[self.CO2] = self.scale_zero_to_max(self.ni_norm[self.CO2])
        #     self.ni_norm[self.CH4] = self.scale_zero_to_max(self.ni_norm[self.CH4])
        #     self.ni_norm[self.H2O] = self.scale_zero_to_max(self.ni_norm[self.H2O])

        #     self.x_CI[self.CO2] = self.scale_zero_to_max(self.x_CI[self.CO2,:])
        #     self.x_CI[self.CH4] = self.scale_zero_to_max(self.x_CI[self.CH4,:])
        #     self.x_CI[self.H2O] = self.scale_zero_to_max(self.x_CI[self.H2O,:])


        # if self.CO2_or_CO_meth == 'CO2':
        #     # Calculate reaction extend of co2 meth
        #     self.conversion_CO2_in_meth = 1- (self.n_CO2_in - self.ni_qss_COx_Phase[self.CH4]) / self.n_CO2_in
             

        #     print(f"CO2 consumption for CH4 prod. (selective): {self.conversion_CO2_in_meth} (calculated with Ch4)") 

        #     self.V_dot_exp_COx_Phase = self.V_dot_Matrix_total+ self.V_dot_str_left_total - (self.V_dot_str_right[self.CO2] + self.V_dot_str_right[self.H2] ) * self.conversion_CO2_in_meth * 2/5  # 1 - 3/5 for volume contraction if 100 conversion 3/5 of original gas remains

        #     plt.figure()
        #     plt.plot(self.t_lin,self.V_out)
        #     plt.xlabel(r'$t$ / s')
        #     plt.axhline(y = self.V_dot_Matrix_total+ self.V_dot_str_left_total, color = 'black',linestyle='-',label = 'exp-H2-Phase' )
        #     plt.axhline(y = self.V_dot_exp_COx_Phase, color = 'black',linestyle='--',label = 'exp-CO2-Phase' )
        #     plt.ylabel(r'$\dot{V}_\mathrm{out}$ / Nml min$^{-1}$')
        #     plt.legend()


        # elif self.CO2_or_CO_meth == 'CO':
        #     # Calculate reaction extend of co meth
        #     self.conversion_CO_in_meth = 1- (self.n_CO_in - self.ni_qss_COx_Phase[self.CH4]) / self.n_CO_in

        #     print(f"CO consumption for CH4 prod. (selective): {self.conversion_CO_in_meth} (calculated with Ch4)") 

        #     self.V_dot_exp_COx_Phase = self.V_dot_Matrix_total+ self.V_dot_str_left_total - (self.V_dot_str_right[self.CO] + self.V_dot_str_right[self.H2] ) * self.conversion_CO_in_meth * 1/2

        #     plt.figure()
        #     plt.plot(self.t_lin,self.V_out)
        #     plt.xlabel(r'$t$ / s')
        #     plt.axhline(y = self.V_dot_Matrix_total+ self.V_dot_str_left_total, color = 'black',linestyle='-',label = 'exp-H2-Phase' )
        #     plt.axhline(y = self.V_dot_exp_COx_Phase, color = 'black',linestyle='--',label = 'exp-CO-Phase' )
        #     plt.ylabel(r'$\dot{V}_\mathrm{out}$ / Nml min$^{-1}$')
        #     plt.legend()

        # plt.figure()
        # plt.plot(self.t_lin,self.ni_out[self.Ar]*1e6,color = self.color[self.Ar] ,label = 'measured @ outlet')
        # plt.axhline(y = self.V_dot_str_right[self.Ar]*self.Nml_min_to_m3_s*self.p0 / self.R / self.T0 *1e6, color = 'black',linestyle='-',label = 'inlet' )
        # plt.xlabel(r'$t$ / s')
        # plt.ylabel(r'$\dot{n}_\mathrm{Ar}$ / $\mathrm{\mu}$mol s$^{-1}$')
        # plt.legend()

        # plt.figure()
        # plt.plot(self.t_lin,self.x_avg_norm[self.CH4,:],'.',label=self.comp[self.CH4], color = self.color[self.CH4]) 
        # plt.plot(self.t_lin,self.x_avg_norm[self.Ar,:],'.',label=self.comp[self.Ar], color = self.color[self.Ar]) 
        # plt.xlabel(r'$t\,/\,\mathrm{s}$')
        # plt.ylabel(r'$x_{i,\mathrm{norm}}\,/\,1$')
        # plt.legend()
        # plt.show()

        # reg_comp = self.CH4
        # plt.figure()
        # plt.fill_between(self.t_lin,y1 = (self.n_avg[reg_comp,:]+self.x_CI[reg_comp,:]*self.n_out_ges) *1e6, y2 = (self.n_avg[reg_comp,:]-self.x_CI[reg_comp,:]*self.n_out_ges) *1e6, color = self.color[reg_comp],alpha=0.5 )
        # plt.plot(self.t_lin,self.n_avg[reg_comp,:]*1e6, color = self.color[reg_comp], zorder=0, label = self.comp[reg_comp] )
        # #print(x_avg[9,:])
        # #plt.xlim((3100,3250))
        # #plt.ylim((1.2,1.4))
        # plt.ylabel(r'$\dot{n}_i\,/\, \mathrm{\mu mol \,s^{-1}}$')
        # plt.xlabel(r'$t\,/\, \mathrm{s}$')
        # plt.legend()
        # plt.show()

        # Comparison theoretical and measured molar flow rates

        # reg = self.CH4
        # #n_out_ges = np.empty_like(n_i[0,:])
        # plt.figure()
        # plt.plot(self.t_lin ,self.ni_RTD[reg,:]*1e6,'k',linestyle='dashdot')
        # plt.plot(self.t_lin ,self.ni_out[reg,:]*1e6,color = self.color[reg])
        # plt.fill_between(self.t_lin ,self.ni_out[reg,:]*1e6,self.ni_RTD[reg,:]*1e6,color='red',alpha=.3)
        # plt.xlabel(r'$t$ / s')
        # plt.ylabel(r'$\dot{n}_\mathrm{CH_4}$ / $\mathrm{\mu}$mol s$^{-1}$')
        # plt.xlim((-20,200))
        # plt.show()

        # plt.figure()
        # plt.plot(self.t_lin,self.ni_trans[reg]*1e6,color = self.color[reg] )
        # plt.xlabel(r'$t$ / s')
        # plt.ylabel(r'$\dot{n}_\mathrm{trans,CH_4}$ / $\mathrm{\mu}$mol s$^{-1}$')
        # plt.legend()

       

        '''
        C and O balance , total adsorped amounts 
        '''
        # Identify components containing carbon up to C3 by screening the comp list
        carbon_components = [i for i, comp_name in enumerate(self.comp) if 'C' in comp_name and ('C4' not in comp_name and 'C5' not in comp_name and 'C6' not in comp_name and 'en' not in comp_name)]
        print(carbon_components)
        # Sum up the transient flows of all C-containing components up to C3
        self.C_trans = np.sum(self.ni_trans[carbon_components, :], axis=0)

        # Identify components containing oxygen by screening the comp list
        oxygen_components = [i for i, comp_name in enumerate(self.comp) if 'O' in comp_name]

        # Sum up the transient flows of all O-containing components
        self.O_trans = np.sum(self.ni_trans[oxygen_components, :], axis=0)
        # Sum up the transient flows of all O-containing components
        self.O_trans = np.sum(self.ni_trans[[self.CO, self.CO2, self.H2O], :], axis=0)

        # Calculate the total amount of C during the H2 Phase via the integral of the first half of C_trans

        # Determine the index for the midpoint of the H2 Phase
        if self.start_H2_Phase:
            midpoint_idx = np.where(self.t_lin > self.t_H2_Phase)[0][0]
        else:
            midpoint_idx = np.where(self.t_lin > self.t_COx_Phase)[0][0]
        print(midpoint_idx)
        print('time for integration split: ',self.t_lin[midpoint_idx])

        if self.CO2_or_CO_meth == 'CO2':
            # conversion for CO2-methanation
            self.X_CO2 = 1 - (self.n_CO2_in - self.ni_qss_COx_Phase[self.CH4]-self.ni_qss_COx_Phase[self.CO]) / self.n_CO2_in
            self.S_CH4_CO2 = self.ni_qss_COx_Phase[self.CH4] / (self.ni_qss_COx_Phase[self.CH4]+self.ni_qss_COx_Phase[self.CO])
            self.S_CO_CO2 = self.ni_qss_COx_Phase[self.CO] / (self.ni_qss_COx_Phase[self.CH4]+self.ni_qss_COx_Phase[self.CO])
            print(f'CO2 conversion: {self.X_CO2}, CH4 selectivity: {self.S_CH4_CO2}, CO selectivity: {self.S_CO_CO2}')
        elif self.CO2_or_CO_meth == 'CO':
            # conversion for CO-methanation
            self.X_CO = 1 - (self.n_CO_in - self.ni_qss_COx_Phase[self.CH4]) / self.n_CO_in
            self.S_CH4_CO = self.ni_qss_COx_Phase[self.CH4] / (self.ni_qss_COx_Phase[self.CH4]+self.ni_qss_COx_Phase[self.CO2])
            self.S_CO2_CO = self.ni_qss_COx_Phase[self.CO2] / (self.ni_qss_COx_Phase[self.CH4]+self.ni_qss_COx_Phase[self.CO2])
            print(f'CO conversion: {self.X_CO}, CH4 selectivity: {self.S_CH4_CO}, CO2 selectivity: {self.S_CO2_CO}')

        # Integrate the first half of C_trans during the H2 Phase
        C_trans_H2_Phase = self.C_trans[midpoint_idx:]
        t_H2_Phase = self.t_lin[midpoint_idx:]

        # Use Simpson's rule to calculate the integral
        self.total_C_H2_Phase = np.trapezoid(y=C_trans_H2_Phase, x=t_H2_Phase) # mol

        print(f"Total amount of C during the first half of the H2 Phase: {self.total_C_H2_Phase:.4e} mol")

        # Calculate the total amount of C during the COx Phase via the integral of the second half of C_trans

        # Determine the index for the midpoint of the COx Phase
        #midpoint_idx_COx = int(self.idx_COx / 2)

        # Integrate the second half of C_trans during the COx Phase
        C_trans_COx_Phase = self.C_trans[:midpoint_idx]
        t_COx_Phase = self.t_lin[:midpoint_idx]

        # Use Simpson's rule to calculate the integral
        self.total_C_COx_Phase = trapezoid(C_trans_COx_Phase, t_COx_Phase) # mol

        print(f"Total amount of C during the second half of the COx Phase: {self.total_C_COx_Phase:.4e} mol")

        # Calculate the total amount of O during the H2 Phase via the integral of the first half of O_trans

        # Integrate the first half of O_trans during the H2 Phase
        O_trans_H2_Phase = self.O_trans[midpoint_idx:]
        t_H2_Phase = self.t_lin[midpoint_idx:]

        # Use Simpson's rule to calculate the integral
        self.total_O_H2_Phase = trapezoid(O_trans_H2_Phase, t_H2_Phase) # mol

        print(f"Total amount of O during the first half of the H2 Phase: {self.total_O_H2_Phase:.4e} mol")

        # Calculate the total amount of O during the COx Phase via the integral of the second half of O_trans

        # Integrate the second half of O_trans during the COx Phase
        O_trans_COx_Phase = self.O_trans[:midpoint_idx]
        t_COx_Phase = self.t_lin[:midpoint_idx]

        # Use Simpson's rule to calculate the integral
        self.total_O_COx_Phase = trapezoid(O_trans_COx_Phase, t_COx_Phase) # mol

        print(f"Total amount of O during the second half of the COx Phase: {self.total_O_COx_Phase:.4e} mol")

        '''
        Calculate the total adsorped amount for each component during the H2 and COx Phases
        '''

        # Initialize arrays to store the results
        self.total_ni_trans_H2_Phase = np.zeros(len(self.comp))
        self.total_ni_trans_COx_Phase = np.zeros(len(self.comp))

        # Loop through each component
        for i in range(self.ni_trans.shape[0]):
            # Transient molar flow rates for the H2 Phase
            ni_trans_H2_Phase = self.ni_trans[i, midpoint_idx:]
            t_H2_Phase = self.t_lin[midpoint_idx:]

            # Use Simpson's rule to calculate the integral
            self.total_ni_trans_H2_Phase[i] = trapezoid(ni_trans_H2_Phase, t_H2_Phase)  # mol

            # Transient molar flow rates for the COx Phase
            ni_trans_COx_Phase = self.ni_trans[i, :midpoint_idx]
            t_COx_Phase = self.t_lin[:midpoint_idx]

            # Use Simpson's rule to calculate the integral
            self.total_ni_trans_COx_Phase[i] = trapezoid(ni_trans_COx_Phase, t_COx_Phase)  # mol

            print(f"Total transient molar amount for {self.comp[i]} during H2 Phase: {self.total_ni_trans_H2_Phase[i]:.4e} mol")
            print(f"Total transient molar amount for {self.comp[i]} during COx Phase: {self.total_ni_trans_COx_Phase[i]:.4e} mol")

    