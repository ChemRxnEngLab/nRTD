import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import glob
from scipy.constants import R
from scipy.integrate import solve_ivp
from datetime import datetime


if 'ICIWstyle' in plt.style.available:
    plt.style.use('ICIWstyle')


def csv_file_finder(snippet, in_a_subfolder):
    """ finds full url by snippet

    Args:
        snippet (string): characteristic snippet of the filename
        in_a_subfolder (string): 'True' or 'False'

    Raises:
        ValueError: more than one result
        ValueError: no result

    Returns:
        string: full url
    """
    if in_a_subfolder=='True':
        files = glob.glob(f'**/*{snippet}*.csv', recursive=True)

    elif in_a_subfolder=='False':
        files = glob.glob(f'*{snippet}*.csv')
        
    if len(files)>1:
        raise ValueError('Not unique key!')
    
    elif len(files)==0:
        raise ValueError('Key not found!')
    
    file_path = files[0]
    
    return file_path


# import section

def csv_to_py(text_file):
    """ text file into numpy array converter

    Args:
        text_file (string): url of text file
        use_column (string): column title

    Returns:
        array: content of text file available as numpy array
    """
    df = pd.read_csv(text_file, delimiter='\t',on_bad_lines='skip') # text file is converted into a pandas DataFrame
    
    arr = df.to_numpy() # pandas DataFrame is converted into an array
     
    return arr

def stoichiometric_matrix(snippet):
    """ stoichiometric matrix and species extraction

    Args:
        snippet (_type_): _description_

    Returns:
        [0]: stoichiometric matrix
        [1]: species
    """

    url = csv_file_finder(snippet,'False') # find the url of the csv file
    csv_entries = csv_to_py(url) # convert csv file into numpy array
   
    # extraction of species
    species = csv_entries[0,0].replace(";", ",").split(",")[1:] # extract species from csv_entries

    # calculation of stoichiometric matrix
    stoichiometric_matrix = [] # create an empty list

    for i in np.arange(1, csv_entries.shape[0]):
        semicolon_replaced_by_comma = csv_entries[i][0].replace(";", ",") # replace ; by , in csv_entries
        results_array = semicolon_replaced_by_comma.split(",") # split the string by ,

        for j in range(len(results_array)):
            if results_array[j] == '':
                results_array[j] = 0 # add zeros into stoichiometric matrix
            else:
                results_array[j] = int(results_array[j]) # convert string into integer

        stoichiometric_matrix.append(results_array[1:]) # remove equation numbering
        
    # return stoichiometric matrix
    stoichiometric_matrix_hin = np.array(stoichiometric_matrix) # convert list into numpy array
    stoichiometric_matrix_zurück = - np.array(stoichiometric_matrix)

    nue = np.vstack((stoichiometric_matrix_hin, stoichiometric_matrix_zurück ))# convert list into numpy array stoichiometric_matrix_hin#

    return nue, species


nue = stoichiometric_matrix('stoichiometric_matrix')[0]

species = stoichiometric_matrix('stoichiometric_matrix')[1]

N_reac = nue.shape[0]
N_species = nue.shape[1]

print(species)
print(nue)


def find_surface_species(species):
    """ returns index of the first surface species

    Args:
        species (list): list of species

    Returns:
        list: index of first surface species
    """
    for i in range(len(species)):
        if species[i].find('s') != -1: # first surface species: index is returned
            return i


def find_support_species(species):
    """ returns index of the first surface species

    Args:
        species (list): list of species

    Returns:
        list: index of first surface species
    """
    for i in range(len(species)):
        if species[i].find('x') != -1: # first surface species: index is returned
            return i


# calculate index of first surface species
surf_start = find_surface_species(species)
support_start = find_support_species(species)

# generate matrices for bulk and surface species
nue_c = nue[:,:surf_start]
nue_s = nue[:,surf_start:support_start]
nue_x = nue[:,support_start:]

# replacing all positive values by 0 and shifting the sign of remaining entries
clean_up = np.vectorize(lambda x: max(0,x))
nue_ss = clean_up(- nue_s)
nue_cc = clean_up(- nue_c)
nue_xx = clean_up(- nue_x)


def rxn(x, theta_s, theta_x, k):
    """ reaction rate function

    Args:
        x (array): gas phase concentrations in 1
        theta_s (array): active material coverage in 1
        theta_x (array): support coverage in 1
        k (float): kinetic constants

    Returns:
        array: reaction rate in mol s-1 m-3
    """

    # creating matrices for c and theta
    x_mat = x.repeat(N_reac).reshape(surf_start, N_reac).T
    theta_s_mat = theta_s.repeat(N_reac).reshape(support_start - surf_start, N_reac).T
    theta_x_mat = theta_x.repeat(N_reac).reshape(N_species - support_start, N_reac).T

    # setting nue_c as the exponent of c
    r = k * np.prod(x_mat**nue_cc, axis=1) * np.prod(theta_s_mat**nue_ss, axis=1) * np.prod(theta_x_mat**nue_xx, axis=1) # calculating reaction rate

    return r 


def mass_balances(t, y, k, n_dot_i_in, V_R, n_max, nx_max, p, T):
    """ mass balance function

    Args:
        t (array): time in s
        y (array): function values in 1
        k (array): kinetic constants in [m, s, mol]
        n_dot_i_in (function): inlet molar flow in mol s-1
        V_R (float): reactor volume in m3
        n_max (float): maximum molar amount of adsorption sites on the active material in mol
        nx_max (float): maximum molar amount of adsorption sites on the support in mol
        p (float): pressure in Pa
        T (float): temperature in K


    Returns:
        array: first order derivatives of y
    """
    # declaration of variables
    x = y[:surf_start]
    theta_s = y[surf_start:support_start]
    theta_x = y[support_start:]

    # calling reaction rate function
    r = rxn(x, theta_s, theta_x, k)
    
    # calculation molar amount within reactor
    n_tot_G = p * V_R / (R * T)

    # calculation of entire outlet molar flow
    n_dot_out = sum(n_dot_i_in(t).ravel()) + V_R * sum(np.matmul(nue_c.T, r))
    
    # mass balances
    dxdt = 1 / n_tot_G * (n_dot_i_in(t).ravel() - x * n_dot_out + V_R * np.matmul(nue_c.T, r)) # mass balance gas phase
    dthetasdt = V_R / n_max * np.matmul(nue_s.T, r) # mass balance catalyst surface
    dthetaxdt = V_R / nx_max * np.matmul(nue_x.T, r) # mass balance support surface
    
    return np.hstack((dxdt, dthetasdt, dthetaxdt)) 


def n_dot_out(sol, k, p, T, n_dot_in, V_R):
    """ total outlet molar flow function

    Args:
        sol (class): solutions of the ODE system
        k (array): kinetic constants
        p (float): pressure in Pa
        T (float): temperature in K
        n_dot_in (float): inlet molar flow in mol s-1
        V_R (float): reactor volume in m3

    Returns:
        array: total outlet molar flow in mol s-1
    """
    t = sol.t
    n_dot_out = np.zeros(t.size) # create an empty array for the outlet molar flow

    for i in np.arange(0, sol.t.size):
        x = sol.y[:surf_start, i]
        theta_s = sol.y[surf_start:support_start, i]
        theta_x = sol.y[support_start:, i]

        # calling reaction rate function
        r = rxn(x, theta_s, theta_x, k)
    
        # calculation of entire outlet molar flow
        n_dot_out[i] = n_dot_in + V_R * sum(np.matmul(nue_c.T, r))

    return n_dot_out


# def solver(k, n_dot_i_in, t_eval, y_init, V_R, n_max, nx_max, p, T):
#     """ solver function

#     Args:
#         k (array): kinetic constants
#         n_dot_i_in (function): inlet molar flow
#         t_eval (array): time points for evaluation
#         y_init (array): initial conditions
#         V_R (float): reactor volume
#         n_max (float): maximum molar amount of adsorption sites on the active material
#         nx_max (float): maximum molar amount of adsorption sites on the support
#         p (float): pressure in Pa
#         T (float): temperature in K

#     Raises:
#         ValueError: integration of ODE system failed

#     Returns:
#         class: solutions of the ODE system
#     """
    
#     res = solve_ivp(mass_balances, 
#                     [t_eval[0], t_eval[-1]], y0 = y_init, t_eval = t_eval,  
#     rtol = 1e-6, #1e-4, 
#     atol = 1e-9, #1e-6, 
#     method = 'Radau', args = ([k, n_dot_i_in, V_R, n_max, nx_max, p, T])) 
    
#     if not res.success == True:
#         raise ValueError('Integration failed!')

#     return res

def solver(k, n_dot_i_in, t_eval, y_init, V_R, n_max, nx_max, p, T, max_step=0.1):
    """ solver function

    Args:
        k (array): kinetic constants
        n_dot_i_in (function): inlet molar flow
        t_eval (array): time points for evaluation
        y_init (array): initial conditions
        V_R (float): reactor volume
        n_max (float): maximum molar amount of adsorption sites on the active material
        nx_max (float): maximum molar amount of adsorption sites on the support
        p (float): pressure in Pa
        T (float): temperature in K

    Raises:
        ValueError: integration of ODE system failed

    Returns:
        class: solutions of the ODE system
    """
    
    res = solve_ivp(mass_balances, 
                    [t_eval[0], t_eval[-1]], y0 = y_init, t_eval = t_eval,  
    rtol = 1e-6, 
    atol = 1e-8, 
    max_step = max_step,
    method = 'Radau', args = ([k, n_dot_i_in, V_R, n_max, nx_max, p, T])) 
    
    if not res.success == True:
        raise ValueError('Integration failed!')

    return res



display('-AZA_solver- imported at',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))