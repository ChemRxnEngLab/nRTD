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


class Output:
    """
        n_dot_func(t)  [prescribed reactor-inlet molar flow, from Reactor_input/F_Ar]
              |
              v
        solve the kinetic ODE (AZA_solver.solver) 
              |
              v
        STAGE E2  (reactor RTD)
              |
              v
        + matrix dilution (H2, Ne after reactor)
              |
              v
        split off one steady-state BU/BT cycle (splitting_function)
              |
              v
        STAGE E3 (x) E4  (outlet piping + capillary)
              |
              v
        n_dot_i(t)
    """

    def __init__(self, ts, Es, n_dot_func, t_eval, y_init, V_R, n_max, nx_max, p, T, surf_start, n_dot_in_bu, t_lin, ni_out_cycle, ni_trans_cycle, ni_RTD_cycle, number_of_periods, switch_delay):
        self.ts = ts                          # time axes of kernels (inlet, reactor, outlet piping, capillary)
        self.Es = Es                          # RTD kernels 
        self.n_dot_func = n_dot_func           # reactor-inlet molar flow per species, mol s-1
        self.t_eval = t_eval                  # time points the ODE solver 
        self.y_init = y_init                  # initial condition for the ODE 
        self.V_R = V_R                        # reactor volume, m3
        self.n_max = n_max                    # maximum molar amount of adsorption sites on the active material, mol
        self.nx_max = nx_max                  # maximum molar amount of adsorption sites on the support, mol
        self.p = p                            # reactor pressure, Pa
        self.T = T                            # reactor temperature, K
        self.surf_start = surf_start          # index of the first surface species in the ODE state vector 
        self.n_dot_in_bu = n_dot_in_bu         # steady-state inlet molar flow per species(BU), mol s-1
        self.t_lin = t_lin                    # experimental time grid 
        self.ni_out_cycle = ni_out_cycle       # measured outlet molar flow per species
        self.ni_trans_cycle = ni_trans_cycle   # measured transient molar flow per species
        self.ni_RTD_cycle = ni_RTD_cycle       # measured RTD response per species
        self.number_of_periods = number_of_periods  
        self.switch_delay = switch_delay       # time offset 

    def extract_RTD(self):

        ts = self.ts
        Es = self.Es

        E1 = Es[0]
        E2 = Es[1]
        E3 = Es[2]
        E4 = Es[3]

        t_1 = ts[0]
        t_2 = ts[1]
        t_3 = ts[2]
        t_4 = ts[3]

        E1 = E1 #/ np.trapezoid(E1, t_1)
        E2 = E2 #/ np.trapezoid(E2, t_2)
        E3 = E3 #/ np.trapezoid(E3, t_3)
        E4 = E4 #/ np.trapezoid(E4, t_4)

        return t_1, t_2, t_3, t_4, E1, E2, E3, E4

    def padding_function(self, kernel, signal, type='None'):

        if type == 'E':
            padding_before = np.zeros(len(kernel) - 1)
            padding_after = np.zeros(len(kernel) - 1)
            padded_solutions = np.hstack((padding_before, signal, padding_after))

        else:
            n_rows = signal.shape[0]
            padding_before = signal[:, 0].repeat(len(kernel) - 1).reshape(n_rows, len(kernel) - 1)
            padding_after = signal[:, -1].repeat(len(kernel) - 1).reshape(n_rows, len(kernel) - 1)
            padded_solutions = np.hstack((padding_before, signal, padding_after))

        return padded_solutions

    def splitting_function(self, solutions, time_per_period, iter, bu_width_fraction=0.5):

        number_of_periods = self.number_of_periods
        x_span = solutions.t

        index_bu = np.where(x_span >= (number_of_periods - 1 - iter) * time_per_period)
        index_bt = np.where(x_span >= (number_of_periods - (1 - bu_width_fraction) - iter) * time_per_period)

        return np.array((index_bu[0][0], index_bt[0][0]))

    def convolution_function(self, k, T):
        t_1 = self.extract_RTD()[0]
        t_2 = self.extract_RTD()[1]
        t_3 = self.extract_RTD()[2]
        t_4 = self.extract_RTD()[3]
        E_1 = self.extract_RTD()[4]
        E_2 = self.extract_RTD()[5]
        E_3 = self.extract_RTD()[6]
        E_4 = self.extract_RTD()[7]

        n_dot_func = self.n_dot_func
        t_eval = self.t_eval
        y_init = self.y_init
        V_R = self.V_R
        n_max = self.n_max
        nx_max = self.nx_max
        p = self.p
        T = self.T
        surf_start = self.surf_start
        n_dot_in_bu = self.n_dot_in_bu
        t_lin = self.t_lin
        ni_out_cycle = self.ni_out_cycle
        ni_trans_cycle = self.ni_trans_cycle
        switch_delay = self.switch_delay

        # Solve the kinetic ODE (mass_balances in solver) 
        solutions = solver(k, n_dot_func, t_eval, y_init, V_R, n_max, nx_max, p, T)

        # Gas-phase mole fractions in idealized reactor
        x_g = solutions.y[:surf_start, :]

        # Active-material and support surface coverages 
        theta_s = solutions.y[surf_start:support_start, :]
        theta_x = solutions.y[support_start:, :]

        # Total outlet molar flow rate leaving the reactor, mol s-1
        n_dot_reactor = n_dot_out(solutions, k, p, T, sum(n_dot_in_bu), V_R)

        # for diagnostics/plotting
        n_dot_species_pre_RTD = x_g * n_dot_reactor[np.newaxis, :]

    
        # STAGE E2 -- reactor RTD for non-ideal mixing/backmixing
        dt2 = t_2[1] - t_2[0]
        t_grid = solutions.t

        n_E2_grid = int(round((t_grid[-1] - t_grid[0]) / dt2)) + 1
        t_E2_grid = np.linspace(t_grid[0], t_grid[-1], n_E2_grid)
        x_g_on_E2_grid = interpolation_function(t_E2_grid, t_grid, x_g)

        x_g_padded = self.padding_function(E_2, x_g_on_E2_grid)

        x_after_E2_on_E2_grid = np.zeros_like(x_g_on_E2_grid)
        for i in np.arange(0, surf_start):
            conv = np.convolve(x_g_padded[i], E_2, mode='valid') * dt2
            x_after_E2_on_E2_grid[i] = conv[:x_g_on_E2_grid.shape[1]]

        # col_sum = x_after_E2_on_E2_grid.sum(axis=0)
        # x_after_E2_on_E2_grid = x_after_E2_on_E2_grid / col_sum[np.newaxis, :]

        x_after_E2 = interpolation_function(t_grid, t_E2_grid, x_after_E2_on_E2_grid)

        # shift n_dot_reactor by E2's mean residence time tau(E2)
        t_2_grid = np.linspace(t_2[0], t_2[-1], len(t_2))  # erzwingt exakt gleichmässige Schrittweite
        dt2_tau = t_2_grid[1] - t_2_grid[0]
        tau_E2 = np.sum(t_2_grid * E_2) * dt2_tau
        n_dot_tot_after_E2 = interpolation_function(t_grid, t_grid + tau_E2, n_dot_reactor)
        
        n_dot_after_E2 = x_after_E2 * n_dot_tot_after_E2[np.newaxis, :]  # absolute flow per species, after E2

        # Matrix dilution: H2 and Ne are dosed downstream of the reactor
        # after reactor
        n_dot_H2_mat = 250 * 1e-6 / 60 * p_STP / (R * T_STP)  # molar flow in mol s-1 (245 mL/min at STP)
        n_dot_Ne_mat = 0 * 1e-6 / 60 * p_STP / (R * T_STP)    # molar flow in mol s-1 (5 mL/min at STP)

        n_dot_after_E2[H2] += n_dot_H2_mat
        n_dot_after_E2[Ne] += n_dot_Ne_mat

        n_dot_tot_after_matrix = n_dot_tot_after_E2 + n_dot_H2_mat + n_dot_Ne_mat

        # Mole fractions after matrix dosing 
        solutions_x_i = n_dot_after_E2 / n_dot_tot_after_matrix[np.newaxis, :]

        index_bu_1, index_bt_1 = self.splitting_function(solutions, 240, 1) #480
        index_bu_2, index_bt_2 = self.splitting_function(solutions, 240, 2) #480

        combined_before_split = np.vstack((solutions_x_i, n_dot_tot_after_matrix[np.newaxis, :]))
        combined_bu = combined_before_split[:, index_bu_2:index_bt_2]   # one BU half-period (120 s)
        combined_bt = combined_before_split[:, index_bt_2:index_bu_1 + 1]  # one BT half-period (360 s)

        
        # STAGE E3 (x) E4 -- outlet piping + capillary
        dt1 = t_1[1] - t_1[0]  # time step of E1 
        dt3 = t_3[1] - t_3[0]  # time step of E3
        dt4 = t_4[1] - t_4[0]  # time step of E4

        if not (dt1 == dt2 == dt3 == dt4):
            raise ValueError("Time steps of RTDs are not equal!")

        # Combine E3 and E4 into a single kernel E_tot = E3 (x) 
        E4_padded = self.padding_function(t_3, E_4, type='E')
        t_tot = np.linspace(0, t_3[-1] + t_4[-1], E_3.size + E_4.size - 1)
        E_tot = np.convolve(E_3, E4_padded, mode='valid') * (t_tot[2] - t_tot[1])
       
        #E_tot = E_tot / np.trapezoid(E_tot, t_tot)  # zwingt E_tot auf Integral=1,
                                              
        t_bu_conv = np.linspace(0, 120, int(120 / (t_tot[1] - t_tot[0])) + 1)
        t_bu_old = np.linspace(0, 120, combined_bu.shape[1])
        t_bt_conv = np.linspace(0, 120, int(120 / (t_tot[1] - t_tot[0])) + 1)
        t_bt_old = np.linspace(0, 120, combined_bt.shape[1])

        combined_bu_int = interpolation_function(t_bu_conv, t_bu_old, combined_bu)  # interpolation of BU phase
        combined_bt_int = interpolation_function(t_bt_conv, t_bt_old, combined_bt)  # interpolation of BT phase

        if t_tot[1] - t_tot[0] != t_bt_conv[1] - t_bt_conv[0] and t_tot[1] - t_tot[0] != t_bu_conv[1] - t_bu_conv[0]:
            raise ValueError('Time steps of RTDs and padded reactor output are not equal!')


        ind_final_bu = t_bu_conv.size
        ind_final_bt = t_bt_conv.size

        n_rows = combined_bu_int.shape[0]  # surf_start (mole fractions) + 1 (total flow)

        padded_combined_bu = self.padding_function(t_tot, combined_bu_int)  # padded BU phase
        padded_combined_bt = self.padding_function(t_tot, combined_bt_int)  # padded BT phase

        combined_in_MS_bu = np.zeros((n_rows, ind_final_bu))
        combined_in_MS_bt = np.zeros((n_rows, ind_final_bt))

        # Convolve every row (all mole fractions + the total-flow row) with E_tot 
        for i in np.arange(0, n_rows):
            bu_results = np.convolve(E_tot, padded_combined_bu[i], mode='valid') * (t_bu_conv[1] - t_bu_conv[0])
            bt_results = np.convolve(E_tot, padded_combined_bt[i], mode='valid') * (t_bt_conv[1] - t_bt_conv[0])

            combined_in_MS_bu[i, :] = bu_results[:ind_final_bu]
            combined_in_MS_bt[i, :] = bt_results[:ind_final_bt]

        # Stitch BU and BT back together into one full cycle 
        combined_in_MS = np.hstack((combined_in_MS_bu, combined_in_MS_bt[:, 1:]))
        t_after_convolution = np.linspace(0, 240, combined_in_MS.shape[1]) #480

        # Split the combined result back into mole fractions and total flow,
        x_final = combined_in_MS[:surf_start, :]
        n_dot_tot_final = combined_in_MS[surf_start, :]

        n_dot_i = x_final * n_dot_tot_final[np.newaxis, :]

        n_dot_theo = theoretical_molar_flow_rates(n_dot_i, ind_final_bu)
        n_dot_trans = transient_molar_flow_rates(n_dot_i, ind_final_bu)

        # Interpolate everything from the internal convolution grid
        n_dot_theo_int = interpolation_function(t_lin, t_after_convolution, n_dot_theo)
        n_dot_trans_int = interpolation_function(t_lin, t_after_convolution, n_dot_trans)
        n_dot_i_int = interpolation_function(t_lin, t_after_convolution, n_dot_i)

        # Carbon/oxygen balance check
        delay = np.where(t_lin > switch_delay)[0][0]

        bu_start_ind = int(t_lin.size / 4) + delay
        C_ads = np.trapezoid(n_dot_trans_int[CH4, bu_start_ind:] + n_dot_trans_int[CO2, bu_start_ind:], t_lin[bu_start_ind:])
        O_ads = np.trapezoid(n_dot_trans_int[H2O, bu_start_ind:] + 2 * n_dot_trans_int[CO2, bu_start_ind:], t_lin[bu_start_ind:])

        diagnostics = {
            't_reactor_grid': solutions.t,                    # actual ODE solver time stamps
            'n_dot_species_pre_RTD': n_dot_species_pre_RTD,    # reactor exit, BEFORE any RTD convolution (pure kinetics)
            'x_after_E2': x_after_E2,                          # mole fractions after E2 (before matrix dilution)
            'n_dot_tot_after_E2': n_dot_tot_after_E2,          # total flow after E2 (before matrix dilution) -- currently just n_dot_reactor, see STAGE E2
            'n_dot_after_E2': n_dot_after_E2,                  # reactor exit AFTER E2 (+ matrix dilution), before E3/E4
            'n_dot_tot_after_matrix': n_dot_tot_after_matrix,  # total flow after E2 + matrix dilution
            'solutions_x_i': solutions_x_i,                    # mole fractions after E2 + dilution
            'n_dot_tot_final': n_dot_tot_final,                # total flow after E3 (x) E4, on t_after_convolution grid
        }

        return {
            'n_dot_theo_int': n_dot_theo_int,
            'n_dot_trans_int': n_dot_trans_int,
            'n_dot_i_int': n_dot_i_int,
            'x_g': x_g,
            'theta_s': theta_s,
            'theta_x': theta_x,
            'C_ads': C_ads,
            'O_ads': O_ads,
            'diagnostics': diagnostics
        }