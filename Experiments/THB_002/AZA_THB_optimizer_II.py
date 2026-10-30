import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import glob
from scipy.constants import R
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution 
from scipy.optimize import basinhopping

import cma

# Import custom modules
from AZA_THB_solver import *
from AZA_THB_tools import *
#from AZA_reactor_output import*
from AZA_THB_reactor_output_II import Output

if 'ICIWstyle' in plt.style.available:
    plt.style.use('ICIWstyle')

count = 0 # start counter
k_list = []
error_list = []
count_list = []
func_opt_list = []

class Optimizer(Output):
    def __init__(self, ts, Es, n_dot_func, t_eval, y_init, V_R, n_max , nx_max, p, T, surf_start, n_dot_in_bu, t_lin, ni_out_cycle, ni_trans_cycle, ni_RTD_cycle, number_of_periods, switch_delay, C_balance, O_balance,
                 cycle_time=240.0, bu_fraction=0.5, periods_from_end=3,
                 V_dot_H2_matrix=250.0, V_dot_Ne_matrix=0.0,
                 t_ht_bu=25.0, t_ht_bt=50.0):
        super().__init__(ts, Es, n_dot_func, t_eval, y_init, V_R, n_max, nx_max, p, T, surf_start,
                         n_dot_in_bu, t_lin, ni_out_cycle, ni_trans_cycle, ni_RTD_cycle,
                         number_of_periods, switch_delay,
                         cycle_time=cycle_time, bu_fraction=bu_fraction,
                         periods_from_end=periods_from_end,
                         V_dot_H2_matrix=V_dot_H2_matrix, V_dot_Ne_matrix=V_dot_Ne_matrix)
        self.switch_delay = switch_delay
        self.C_balance = C_balance
        self.O_balance = O_balance
        self.t_ht_bu = float(t_ht_bu)   # Dauer des hochtransienten BU-Fensters, s
        self.t_ht_bt = float(t_ht_bt)   # Dauer des hochtransienten BT-Fensters, s

    def weighted_mean_squared_deviation(self, exp_data, sim_data, species, trans):
        t_lin = self.t_lin
        switch_delay = self.switch_delay

        delay = np.where(t_lin > switch_delay)[0][0] # find index of switch delay

        dt_lin = t_lin[1] - t_lin[0]

        # BU->BT-Umschaltpunkt. War: int(t_lin.size * 1/4) -> bei 240 s Periode
        # 60 s statt 120 s, d.h. alle Gewichtungsfenster sassen an der falschen
        # Stelle und die echte BT-Flanke fiel mitten in das bu_ss-Fenster.
        split_at = int(round(self.bu_fraction * self.cycle_time / dt_lin))

        # Fensterlaengen in Sekunden statt in Stuetzstellen (vorher 50/100 Punkte,
        # auf dem 506-Punkte-Gitter 23.8 s / 47.5 s). Fuer exakt die alte
        # Gewichtung: t_ht_bu=23.76, t_ht_bt=47.52 setzen.
        len_ht_bu = int(round(self.t_ht_bu / dt_lin))
        len_ht_bt = int(round(self.t_ht_bt / dt_lin))
        

        if species =='CH4':
            corr_1 = ((exp_data - sim_data) / max(exp_data))**2 # normalize error to maximum of experimental data
            
        elif species == 'CO2':
            corr_1 = ((exp_data - sim_data) / max(exp_data))**2 # normalize error to maximum of experimental data

        elif species == 'H2O':
            corr_1 = ((exp_data - sim_data) / max(exp_data))**2 # normalize error to maximum of experimental data

         # putting special emphasis on highly transient regime
        if trans == True:
            corr_bu_max_trans = corr_1[ delay : len_ht_bu + delay] * 2e5  
            corr_bu_ss        = corr_1[ len_ht_bu + delay : split_at + delay] * 0.5e5
            corr_bt_max_trans = corr_1[ split_at + delay : split_at + len_ht_bt + delay] * 1e5
            corr_bt_ss        = np.hstack((corr_1[ split_at + len_ht_bt + delay : ], corr_1[ : delay])) * 1e3

        elif trans == False:
            corr_bu_max_trans = corr_1[ delay : len_ht_bu + delay] * 0.5e4  
            corr_bu_ss        = corr_1[ len_ht_bu + delay : split_at + delay] * 2e5
            corr_bt_max_trans = corr_1[ split_at + delay : split_at + len_ht_bt + delay] * 0.5e4
            corr_bt_ss        = np.hstack((corr_1[ split_at + len_ht_bt + delay : ], corr_1[ : delay])) * 5e4

        # stack time periods together
        corr_3 = np.hstack((corr_bu_max_trans, 
                            corr_bu_ss,
                            corr_bt_max_trans,
                            corr_bt_ss))


        if len(count_list) == 0:
            plt.plot(t_lin[delay:], corr_3[:-(delay)]/max(corr_3), label='error', color='crimson', linestyle = '--')
            plt.plot(t_lin[delay:], sim_data[delay:]/max(sim_data), label='sim')
            plt.plot(t_lin[delay:], exp_data[delay:]/max(exp_data), label='exp')
            plt.xlabel('t / s')
            plt.ylabel('Deviation')
            plt.legend()
            plt.show()

            plt.plot(t_lin, exp_data, label='exp')
            plt.plot(t_lin, sim_data, label='sim')
            plt.axvline(x=t_lin[delay], color='gray', linestyle='--')
            plt.axvline(x=t_lin[len_ht_bu + delay], color='gray', linestyle='--')
            plt.axvline(x=t_lin[split_at + delay], color='gray', linestyle='--')
            plt.axvline(x=t_lin[split_at + len_ht_bt + delay], color='gray', linestyle='--')
            plt.legend()
            plt.show()

        if len(corr_3) != t_lin.size:
            raise ValueError("Length of correlation array does not match length of time array.")

        return corr_3


    def objective_function(self, S):
        t_lin = self.t_lin
        T = self.T

        # experimental data
        n_dot_trans_exp = self.ni_trans_cycle
        n_dot_exp = self.ni_out_cycle
        ni_RTD_cycle = self.ni_RTD_cycle
        C_ads_exp = self.C_balance
        O_ads_exp = self.O_balance

        global count
    
        k_sim = np.abs(S)

        # digital twin output
        sim_res = self.convolution_function(k_sim, T)
        n_dot_trans_sim = sim_res['n_dot_trans_int']
        n_dot_sim = sim_res['n_dot_i_int']
        C_ads_sim = sim_res['C_ads']
        O_ads_sim = sim_res['O_ads']

    
        # calculating weighted deviation to experimental data 
        # Hinweis: die Gewichte summieren sich zu 10, der Vorfaktor 1/6 ist ein
        # Erbstueck. Reine Skalierung, verschiebt das Optimum nicht.
        dev = 1/6 * (#np.average(self.weighted_mean_squared_deviation(n_dot_trans_exp[CO2], n_dot_trans_sim[CO2, :], 'CO2', trans = True))
                    + np.average(self.weighted_mean_squared_deviation(n_dot_trans_exp[CH4], n_dot_trans_sim[CH4, :], 'CH4', trans = True)) * 6
                    + np.average(self.weighted_mean_squared_deviation(n_dot_trans_exp[H2O], n_dot_trans_sim[H2O, :], 'H2O', trans = True))

                    #+ np.average(self.weighted_mean_squared_deviation(n_dot_exp[CO2], n_dot_sim[CO2, :], 'CO2', trans = False))
                    + np.average(self.weighted_mean_squared_deviation(n_dot_exp[CH4], n_dot_sim[CH4, :], 'CH4', trans = False)) * 2
                    + np.average(self.weighted_mean_squared_deviation(n_dot_exp[H2O], n_dot_sim[H2O, :], 'H2O', trans = False))
                    
                    #  + np.abs(((C_ads_exp - C_ads_sim) / C_ads_exp)**2) * 2000
                    #  + np.abs(((O_ads_exp - O_ads_sim) / O_ads_exp)**2) * 1000
                    ) 
        
        print('k_sim',k_sim)
        print('dev',dev)
        count+=1
        k_list.append(k_sim)
        error_list.append(dev)
        count_list.append(count)

        print('i',count)

       
        return dev

    def optimizer(self, initial_guess):
        if len(count_list) != 0:
            raise ValueError("Please clear all outputs.")
        
        # call optimization algorithm
        k_opt = basinhopping(self.objective_function,
                                initial_guess,  # initial guess
                                T = 10,  # temperature criterion
                                niter = 5, # number of iterations 
                                stepsize = 50,# stepsize
                                minimizer_kwargs = {'method':'Nelder-Mead','tol' : 1,"options": {"disp": True, "adaptive":True}, 'bounds': [(0, 1e14)] * len(initial_guess)}, # select minimizer and properties
                                stepwise_factor = 1e-4
                                )
        
        return k_opt

display('-AZA_THB_optimizer- imported at',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))