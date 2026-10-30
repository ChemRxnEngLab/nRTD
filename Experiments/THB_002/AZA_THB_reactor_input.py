import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import glob
from scipy.constants import R
from scipy.integrate import solve_ivp

# Import custom modules
from AZA_THB_solver import *
from AZA_THB_tools import *

if 'ICIWstyle' in plt.style.available:
    plt.style.use('ICIWstyle')


class Reactor_input:
    def __init__(self, x_i_RTD, t_RTD, t_lin):
        self.x_i_RTD = x_i_RTD
        self.t_RTD = t_RTD
        self.t_lin = t_lin


    def extract_half_periods(self):
        x_i_RTD = self.x_i_RTD
        n_total = x_i_RTD.shape[0]
        n_periods = n_total // 4
        x_Ar_bu = []
        x_Ar_bt = []
        x_He_bu = []
        x_He_bt = []

        for i in range(n_periods):
            x_Ar_bu.append(x_i_RTD[    4 * i, 0, ...])
            x_He_bt.append(x_i_RTD[1 + 4 * i, 0, ...])
            x_Ar_bt.append(x_i_RTD[2 + 4 * i, 0, ...])
            x_He_bu.append(x_i_RTD[3 + 4 * i, 0, ...])

        return np.array(x_Ar_bu), np.array(x_Ar_bt), np.array(x_He_bu), np.array(x_He_bt)
    

    def averaging_half_periods(self):
        
        x_Ar_bu = self.extract_half_periods()[0]
        x_Ar_bt = self.extract_half_periods()[1]
        x_He_bu = self.extract_half_periods()[2]
        x_He_bt = self.extract_half_periods()[3]
       
        x_Ar_bu_avg = np.average(x_Ar_bu, axis=0)
        x_Ar_bt_avg = np.average(x_Ar_bt, axis=0)
        x_He_bu_avg = np.average(x_He_bu, axis=0)
        x_He_bt_avg = np.average(x_He_bt, axis=0)

        return x_Ar_bu_avg, x_Ar_bt_avg, x_He_bu_avg, x_He_bt_avg 


    def stacking(self):

        x_Ar_avg = np.hstack((self.averaging_half_periods()[0], self.averaging_half_periods()[1][1:] ))
        x_He_avg = np.hstack((self.averaging_half_periods()[2], self.averaging_half_periods()[3][1:] ))
        
        return x_Ar_avg, x_He_avg


    def interpolating(self):
        t_lin = self.t_lin
        x_Ar_avg, x_He_avg = self.stacking()
        
        t_old = np.linspace(0, 240, x_Ar_avg.size) 
        x_Ar_avg_int = interpolation_function(t_lin, t_old, x_Ar_avg) # calculate average Ar response
        x_He_avg_int = interpolation_function(t_lin, t_old, x_He_avg) # calculate average He response
        plt.title('Ar')
        plt.plot(t_lin, x_Ar_avg_int, label='Ar')
        plt.title('He')
        plt.plot(t_lin, x_He_avg_int, label='He')
        plt.title('total')
        plt.plot(t_lin, x_Ar_avg_int + x_He_avg_int, label='Ar')
        return x_Ar_avg_int, x_He_avg_int   
    
     
    # def F_Ar(self):
    #     Ar_interpolated = self.interpolating()[0]
        
    #     end_of_bu_period = int(Ar_interpolated.size / 4)
    #     x_qss_up = np.mean(Ar_interpolated[end_of_bu_period - 100 : end_of_bu_period]) # quasi steady state concentrations for bu phase are extracted   #-100
    #     x_qss_down = np.mean((Ar_interpolated[ -100 : -1])) # quasi steady.state values for bt phase are extracted  #-100

    #     F_Ar = (Ar_interpolated - x_qss_down) / (x_qss_up - x_qss_down) # calculate cumulative residence time distribution
        
    #     return F_Ar
    
    def F_Ar(self):
        Ar_interpolated = self.interpolating()[0]

    # FIX: robustes Plateau-Fenster statt hartkodiertem size/4 (Erbstück der alten
    # 480s/BU=1/4-Konvention). Nimmt das Maximum der Kurve selbst als Ankerpunkt.
        peak_idx = np.argmax(Ar_interpolated)
        x_qss_up = np.mean(Ar_interpolated[max(0, peak_idx - 20) : peak_idx + 1])
        x_qss_down = np.mean(Ar_interpolated[-20 : -1])

        F_Ar = (Ar_interpolated - x_qss_down) / (x_qss_up - x_qss_down)
        return F_Ar

display('-AZA_reactor_input- imported at',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        