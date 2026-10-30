import numpy as np
from datetime import datetime

# Creation of general variables
comp = np.array(['Ar','He','H2','CO','CO2','CH4','H2O','C2H6','C3H8','Ne','C4','C5','C6','C2en','C3en']) # Component names

Ar = np.where(comp=='Ar')[0][0]
He = np.where(comp=='He')[0][0]
H2 = np.where(comp=='H2')[0][0]
CO = np.where(comp=='CO')[0][0]
CO2 = np.where(comp=='CO2')[0][0]
CH4 = np.where(comp=='CH4')[0][0]
H2O = np.where(comp=='H2O')[0][0]
C2 = np.where(comp=='C2H6')[0][0]
C3 = np.where(comp=='C3H8')[0][0]
Ne = np.where(comp=='Ne')[0][0]
C4 = np.where(comp=='C4')[0][0]
C5 = np.where(comp=='C5')[0][0]
C2en = np.where(comp=='C2en')[0][0]
C3en = np.where(comp=='C3en')[0][0]

# give each component a individual color
clr_comp = np.full(len(comp),'empty color')
clr_comp[Ar] = 'cyan'
clr_comp[H2] = 'green'
clr_comp[He] = 'grey'
clr_comp[CO] = 'red'
clr_comp[CH4] = 'blue'
clr_comp[C2] = 'mediumblue'
clr_comp[C3] = 'royalblue'
clr_comp[C4] = 'deepskyblue'
clr_comp[C5] = 'skyblue'
clr_comp[H2O] = 'y'
clr_comp[Ne] = 'brown'


# Standard conditions
p_STP = 101325 # standard pressure in Pa
T_STP = 273.15 # standard temperature in K


def interpolation_function(x_new, x_old, y_old):
    """ interpolates arbitrary datasets over arbitrary x-values

    Args:
        x_new (array): x values of the new dataset
        x_old (array): x values old dataset
        y_old (array): y values old dataset

    Returns:
        array: y values over x_new
    """
    a = 'many'
    b = 'many'
    # y_old multidimensional?
    try:
        try_1 = y_old.shape[1]  
    except AttributeError:
        a = 'one'
    except IndexError:
        a = 'one' 

    # x_new multidimensional?
    try:
        try_2 = x_new.size
    except AttributeError:
        b = 'one'

    if a == 'one' and b =='one':
        y = np.interp(x_new, x_old, y_old)


    elif a=='many' and b=='one':
        y = np.zeros((y_old.shape[0]))
        for i in np.arange(0, y_old.shape[0]):
            y[i] = np.interp(x_new, x_old, y_old[i])

    elif a=='many' and b=='many':
        y = np.zeros((y_old.shape[0], x_new.size))
        for i in np.arange(0, y_old.shape[0]):
            y[i,:] = np.interp(x_new, x_old, y_old[i,:])

    elif a=='one' and b=='many':
        y = np.interp(x_new, x_old, y_old)
    
    return y


def F_Ar(n_dot, ind_final):
    """ Cumulative residence time distribution

    Args:
        n_dot (array): molar flow rates of all components
        ind_final (int): index after period

    Returns:
        array: cumulative residence time distribution
    """
    
    n_dot_qss_up = np.mean(n_dot[Ar, ind_final - 100 : ind_final])
    n_dot_qss_down = np.min((n_dot[Ar, :]))
    F_ar = (n_dot[Ar] - n_dot_qss_down) / (n_dot_qss_up - n_dot_qss_down) # calculate cumulative residence time distribution
    
    return F_ar


def theoretical_molar_flow_rates(n_dot, ind_final):
    """ Theoretical molar flow rates

    Args:
        n_dot (array): molar flow rates of all components
        ind_final (int): index after period

    Returns:
        array: theoretical molar flow rates
    """

    F_ar = F_Ar(n_dot, ind_final) 

    n_dot_theo = np.zeros((n_dot.shape[0], F_ar.size))

    n_dot_bu = n_dot[:, :ind_final]
    n_dot_bt = n_dot[:, ind_final:]

    for i in np.arange(0, n_dot.shape[0]):
        n_dot_bu_max = np.mean(n_dot_bu[i, ind_final-100:ind_final])
        n_dot_bt_min = np.min(n_dot_bu[i,  : ])
        n_dot_theo[i] = F_ar * (n_dot_bu_max - n_dot_bt_min) + n_dot_bt_min # calculated theoretical molar flow rates

    return n_dot_theo


def transient_molar_flow_rates(n_dot, ind_final):
    """ Transient molar flow rates

    Args:
        n_dot (array): molar flow rates of all components
        ind_final (int): index after period

    Returns:
        array: transient molar flow rates
    """

    n_dot_theo = theoretical_molar_flow_rates(n_dot, ind_final)

    n_dot_trans = n_dot - n_dot_theo # calculate transient molar flow rates

    return n_dot_trans

display('-AZA_fancy_tools- imported at',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
