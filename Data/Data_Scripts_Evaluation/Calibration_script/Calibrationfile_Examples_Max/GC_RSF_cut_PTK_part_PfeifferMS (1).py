# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:20:04 2024

@author: MaxGäßler

Import MS Data from Pfeiffer MS ascii files
"""

import numpy as np
from tkinter import filedialog,messagebox
from tkinter.messagebox import askquestion
from tkinter.simpledialog import askstring,askinteger
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os


print('Calculating the RSF matrix...')
input_filepaths = filedialog.askopenfilenames(initialdir=r'C:\Users\MaxGäßler\Documents\Rohdaten',title='Select Calibration Files')
# input_filepaths = [r'C:\Users\simon\OneDrive - bwedu\Studium\02_Master\01_Masterarbeit\02_Versuche\03_Rohdaten\MS Kalibrierung\22_10\001_0Ar_016_1He_129_2H2_009_0CH4_005_0Ne_MASimon_20221025125626.txt',
#                     r'C:\Users\simon\OneDrive - bwedu\Studium\02_Master\01_Masterarbeit\02_Versuche\03_Rohdaten\MS Kalibrierung\22_10\001_0Ar_016_1He_129_2H2_009_0CO_005_0Ne_MASimon_20221025100556.txt']
counter = 0
for path_cal in input_filepaths:
    print('Used calibraion file:',path_cal)
    ms_header_cal = np.loadtxt(path_cal,delimiter='\t',skiprows=6,usecols=(2,5,8,11,14,17,20,23,26,29,32),dtype=str,max_rows=1)
    ms_header_cal = np.char.replace(ms_header_cal, '"', '')
    ms_data_cal = np.loadtxt(path_cal,delimiter='\t',skiprows=8,usecols=(2,5,8,11,14,17,20,23,26,29,32))
    ms_data_avg_cal = np.mean(ms_data_cal,axis=0)
    ms_data_std_cal = np.std(ms_data_cal,axis=0)
    if len(ms_header_cal) != len(ms_data_avg_cal):
        raise ValueError('Header and data don´t fit!')

    ## Bestimmung der Konzentrationen aus Kaibrierdatei
    comp = np.array(['time','Ar','He','H2','CO','CO2', 'CH4','H2O','C2','C3','Ne','C4','C5','C6','C2en','C3en'])
    # indx =        [0,     1,    2,   3,    4,     5,    6,   7,   8,   9,  10,  11,  12,  13] # component indices

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
                    matrix_corr = 1.2 # Correction of MFC "H2 Matrix"
                    H2_matrix = float(filename[ind_matrix-5:ind_matrix-2]) + float(filename[ind_matrix-1])/10
                    Vdot[i] = Vdot[i]+H2_matrix*matrix_corr
                    print('H2 Matrix:',H2_matrix,'*',matrix_corr,'=',H2_matrix*matrix_corr)
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
    x_ref = x_i_cal[np.where(comp=='He')[0][0]] # x_ref = x_He as reference
    Idot_ref = ms_data_avg_cal[np.where(ms_header_cal=='He')[0][0]] # Idot_ref = Mass 4
    ion_ref = He_ion

    # Choose where to evaluate the RSF values
    eval_at = np.zeros((len(comp),len(ms_data_avg_cal))) # if eval_at is one at a position this combination is evaluated regarding the RSF(component,mass)
    eval_at[np.where(comp=='Ar')[0][0],np.where(ms_header_cal=='Ne')[0][0]] = 1 # Ar,20
    eval_at[np.where(comp=='Ar')[0][0],np.where(ms_header_cal=='Ar')[0][0]] = 1 # Ar,40
    eval_at[np.where(comp=='He')[0][0],np.where(ms_header_cal=='He')[0][0]] = 1 # He,4
    eval_at[np.where(comp=='H2')[0][0],np.where(ms_header_cal=='H2')[0][0]] = 1 # H2,2
    eval_at[np.where(comp=='CO')[0][0],np.where(ms_header_cal=='CO')[0][0]] = 1 # CO,28
    eval_at[np.where(comp=='CH4')[0][0],np.where(ms_header_cal=='CH4')[0][0]] = 1 # CH4,15
    # eval_at[np.where(comp=='CH4')[0][0],np.where(ms_header_cal=='Mass 16')[0][0]] = 1 # CH4,16
    # eval_at[np.where(comp=='H2O')[0][0],np.where(ms_header_cal=='Mass 16')[0][0]] = 1 # H2O,16
    eval_at[np.where(comp=='H2O')[0][0],np.where(ms_header_cal=='H2O')[0][0]] = 1 # H2O,18
    # eval_at[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='Mass 30')[0][0]] = 1 # C2,30
    # eval_at[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='Mass 28')[0][0]] = 1 # C2,28
    # eval_at[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='Mass 28')[0][0]] = 1 # C3,28
    # eval_at[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='Mass 43')[0][0]] = 1 # C3,43
    eval_at[np.where(comp=='Ne')[0][0],np.where(ms_header_cal=='Ne')[0][0]] = 1 # Ne,20

    # calculate RSF matrix
    RSF_temp = np.zeros_like(eval_at)
    for mm in range(eval_at.shape[1]):
        for i in range(len(x_i_cal)):
            if x_i_cal[i] != 0: # only calculate RSF if component is present in the calibration measurement
                RSF_temp[i,mm] = x_ref/x_i_cal[i]*ms_data_avg_cal[mm]/Idot_ref*eval_at[i,mm]

    if path_cal == input_filepaths[0]:
        RSF_all = np.array([RSF_temp])
    else:
        RSF_all = np.concatenate((RSF_all,np.array([RSF_temp])),axis=0)
    
    print()
    print('RSF',counter,'in %:\n',ms_header_cal,'\n',np.array(RSF_temp*100,dtype=int))
    counter += 1

# Calculate final RSF matrix
# Take the mean of values that occur more than one time
RSF = np.true_divide(RSF_all.sum(0),(RSF_all!=0).sum(0))
# RSF = np.nanmean(np.where(RSF_all!=0,RSF_all,np.nan),0)
RSF = np.nan_to_num(RSF,nan=0.,posinf=0.,neginf=0.)

# Use the literature values for all non calibrated components
# Reference: Cracking Patterns
non_calib = ~RSF.any(axis=1)
# non_calib = np.full((len(comp)),True) # uncomment to show the literatur data
if non_calib[np.where(comp=='Ar')[0][0]]: # Ar not calibrated
    RSF[np.where(comp=='Ar')[0][0],np.where(ms_header_cal=='Ne')[0][0]] = 1*Ar_ion/ion_ref # Ar,20
    RSF[np.where(comp=='Ar')[0][0],np.where(ms_header_cal=='Ar')[0][0]] = 0.1*Ar_ion/ion_ref # Ar,40
if non_calib[np.where(comp=='He')[0][0]]: # He not calibrated
    RSF[np.where(comp=='He')[0][0],np.where(ms_header_cal=='He')[0][0]] = 1*He_ion/ion_ref
if non_calib[np.where(comp=='H2')[0][0]]: # H2 not calibrated
    RSF[np.where(comp=='H2')[0][0],np.where(ms_header_cal=='H2')[0][0]] = 1*H2_ion/ion_ref
if non_calib[np.where(comp=='CO')[0][0]]: # CO not calibrated
    RSF[np.where(comp=='CO')[0][0],np.where(ms_header_cal=='CO')[0][0]] = 1*CO_ion/ion_ref
if non_calib[np.where(comp=='CH4')[0][0]]: # CH4 not calibrated
    RSF[np.where(comp=='CH4')[0][0],np.where(ms_header_cal=='CH4')[0][0]] = 0.85*CH4_ion/ion_ref # CH4 at Mass 15
    #RSF[np.where(comp=='CH4')[0][0],np.where(ms_header_cal=='Mass 16')[0][0]] = 1*CH4_ion/ion_ref # CH4 at Mass 16
if non_calib[np.where(comp=='H2O')[0][0]]: # H2O not calibrated
   # RSF[np.where(comp=='H2O')[0][0],np.where(ms_header_cal=='Mass 16')[0][0]] = 0.02*H2O_ion/ion_ref # H2O at Mass 16
    RSF[np.where(comp=='H2O')[0][0],np.where(ms_header_cal=='H2O')[0][0]] = 1*H2O_ion/ion_ref # H2O at Mass 18
# if non_calib[np.where(comp=='C2')[0][0]]: # C2 not calibrated
#     RSF[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='Mass 30')[0][0]] = 0.26*C2H6_ion/ion_ref # C2 at Mass 30
#     RSF[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='Mass 28')[0][0]] = 1*C2H6_ion/ion_ref  # C2 at Mass 28
# if non_calib[np.where(comp=='C3')[0][0]]: # C3 not calibrated
#     RSF[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='Mass 28')[0][0]] = 0.59*C3H8_ion/ion_ref # C3 at Mass 28
#     RSF[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='Mass 43')[0][0]] = 0.22*C3H8_ion/ion_ref # C3 at Mass 43
if non_calib[np.where(comp=='Ne')[0][0]]: # Ne not calibrated
    RSF[np.where(comp=='Ne')[0][0],np.where(ms_header_cal=='Ne')[0][0]] = 1*Ne_ion/ion_ref

print()
print('RSF final in %:\n',np.array(RSF*100,dtype=int),'\n')

messagebox.showinfo(message='RSF matrix calculation finished!')
#==================================================================

# calculate the composition

print('Calculating the compositions of the given experiment...')

path_exp = filedialog.askopenfilename(initialdir=r'C:\Users\MaxGäßler\Documents\Rohdaten\MS',title='Select MS File')
# path_exp = r'C:\Users\simon\OneDrive - bwedu\Studium\02_Master\01_Masterarbeit\02_Versuche\03_Rohdaten\MS\SHA-E-001_Vorversuch_p1_20220923093041.txt'
# path = input_filepaths[0]
print(path_exp)
ms_header_exp = np.loadtxt(path_exp,delimiter='\t',skiprows=6,usecols=(2,5,8,11,14,17,20,23,26,29,32),dtype=str,max_rows=1)
ms_header_exp = np.char.replace(ms_header_exp, '"', '')
ms_data_exp = np.loadtxt(path_exp,delimiter='\t',skiprows=8,usecols=(2,5,8,11,14,17,20,23,26,29,32))
# read the measurement time
print('Reading the measurement time...')
ms_time_exp = np.loadtxt(path_exp,delimiter='\t',skiprows=8,usecols=(0,3,6,9,12,15,18,21,24,27,30),dtype=str)
ms_time_elap = np.loadtxt(path_exp,delimiter='\t',skiprows=8,usecols=(1,4,7,10,13,16,19,22,25,28,31))

# extract evaluated times
ms_time_exp = np.char.strip(np.char.add('0',ms_time_exp))
#ms_time_exp = np.char.add('"',ms_time_exp)
#ms_time_exp = np.char.add(ms_time_exp,'"')
pd_times = pd.to_datetime(ms_time_exp[:,0],format='%m.%d.%Y %H:%M:%S.%f')
ms_times = np.array(pd_times,dtype=np.datetime64)
# ms_time_elap = (ms_times-ms_times[0])/ np.timedelta64(1, 's')

# Correction of the influence of consecutive masses in ms measurements
fig,ax = plt.subplots()
# ax.plot(ms_time_elap,ms_data_exp[:,np.where(ms_header_exp=='Mass 4')[0][0]+1],label='Mass 43, vorher')
ax.plot(ms_time_elap,ms_data_exp[:,np.where(ms_header_exp=='Ne')[0][0]+1],label='C4, vorher')
# ms_data_exp[:,np.where(ms_header_exp=='Mass 4')[0][0]+1] = ms_data_exp[:,np.where(ms_header_exp=='Mass 4')[0][0]+1] - 0.12/100*ms_data_exp[:,np.where(ms_header_exp=='Mass 4')[0][0]]
ms_data_exp[:,np.where(ms_header_exp=='Ne')[0][0]+1] = ms_data_exp[:,np.where(ms_header_exp=='Ne')[0][0]+1] - 0.89/100*ms_data_exp[:,np.where(ms_header_exp=='Ne')[0][0]]
# ax.plot(ms_time_elap,ms_data_exp[:,np.where(ms_header_exp=='Mass 4')[0][0]+1],label='Mass 43, nachher')
ax.plot(ms_time_elap,ms_data_exp[:,np.where(ms_header_exp=='Ne')[0][0]+1],label='C4, nachher')
ax.legend()
plt.show()



# Caclulate each component
# Argon
nt_Ar = ms_data_exp[:,np.where(ms_header_exp=='Ar')[0][0]]/RSF[np.where(comp=='Ar')[0][0],np.where(ms_header_cal=='Ar')[0][0]]
# Helium
nt_He = ms_data_exp[:,np.where(ms_header_exp=='He')[0][0]]/RSF[np.where(comp=='He')[0][0],np.where(ms_header_cal=='He')[0][0]]
# Hydrogen
nt_H2 = ms_data_exp[:,np.where(ms_header_exp=='H2')[0][0]]/RSF[np.where(comp=='H2')[0][0],np.where(ms_header_cal=='H2')[0][0]]
# Methane
nt_CH4 = ms_data_exp[:,np.where(ms_header_exp=='CH4')[0][0]]/RSF[np.where(comp=='CH4')[0][0],np.where(ms_header_cal=='CH4')[0][0]]
# Water
nt_H2O = ms_data_exp[:,np.where(ms_header_exp=='H2O')[0][0]]/RSF[np.where(comp=='H2O')[0][0],np.where(ms_header_cal=='H2O')[0][0]]
# Ethane
# nt_C2 = ms_data_exp[:,np.where(ms_header_exp=='Mass 30')[0][0]]/RSF[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='Mass 30')[0][0]]
nt_C2 = np.zeros_like(nt_Ar)
# print(RSF[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='Mass 30')[0][0]])
# Propane
# nt_C3 = ms_data_exp[:,np.where(ms_header_exp=='Mass 43')[0][0]]/RSF[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='Mass 43')[0][0]]
# nt_C3 = ms_data_exp[:,np.where(ms_header_exp=='Mass 44')[0][0]]/RSF[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='Mass 44')[0][0]]
nt_C3 = np.zeros_like(nt_Ar)
# CO
Idot28 = ms_data_exp[:,np.where(ms_header_exp=='CO')[0][0]]
# Idot30 = ms_data_exp[:,np.where(ms_header_exp=='Mass 30')[0][0]]
# Idot26 = ms_data_exp[:,np.where(ms_header_exp=='Mass 26')[0][0]]
# Idot43 = ms_data_exp[:,np.where(ms_header_exp=='Mass 43')[0][0]]
RSF_C2_28 = RSF[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='CO')[0][0]]
# RSF_C2_30 = RSF[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='Mass 30')[0][0]]
# RSF_C2_26 = RSF[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='Mass 26')[0][0]]
RSF_C3_28 = RSF[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='CO')[0][0]]
# RSF_C3_43 = RSF[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='Mass 43')[0][0]]
RSF_CO_28 = RSF[np.where(comp=='CO')[0][0],np.where(ms_header_cal=='CO')[0][0]]
# if gc_prompt == 'yes':
#     nt_CO = Idot28/RSF_CO_28 # without C2 and C3
# else:
#     nt_CO = (Idot28 - Idot30*RSF_C2_28/RSF_C2_30 - Idot43*RSF_C3_28/RSF_C3_43)/RSF_CO_28 # with C2 and C3
nt_CO = Idot28/RSF_CO_28 # without C2 and C3
# Ne
Idot20 = ms_data_exp[:,np.where(ms_header_exp=='Ne')[0][0]]
Idot40 = ms_data_exp[:,np.where(ms_header_exp=='Ar')[0][0]]
RSF_Ar_20 = RSF[np.where(comp=='Ar')[0][0],np.where(ms_header_cal=='Ne')[0][0]]
RSF_Ar_40 = RSF[np.where(comp=='Ar')[0][0],np.where(ms_header_cal=='Ar')[0][0]]
RSF_Ne_20 = RSF[np.where(comp=='Ne')[0][0],np.where(ms_header_cal=='Ne')[0][0]]
# nt_Ne = (Idot20 - Idot40*RSF_Ar_20/RSF_Ar_40)/RSF_Ne_20 # With Argon
nt_Ne = Idot20/RSF_Ne_20 # without argon
# CO2
nt_CO2 = np.zeros_like(nt_Ar)
# C4
nt_C4 = np.zeros_like(nt_Ar)
# C5
nt_C5 = np.zeros_like(nt_Ar)
# C6
nt_C6 = np.zeros_like(nt_Ar)

# All in one array
# nt_all = np.zeros((len(nt_Ar),len(comp)))
# nt_all[:,np.where(comp=='Ar')[0][0]] = nt_Ar
# nt_all[:,np.where(comp=='He')[0][0]] = nt_He
# nt_all[:,np.where(comp=='H2')[0][0]] = nt_H2
# nt_all[:,np.where(comp=='CO')[0][0]] = nt_CO
# nt_all[:,np.where(comp=='CH4')[0][0]] = nt_CH4
# nt_all[:,np.where(comp=='H2O')[0][0]] = nt_H2O
# nt_all[:,np.where(comp=='C2')[0][0]] = nt_C2
# nt_all[:,np.where(comp=='C3')[0][0]] = nt_C3
# nt_all[:,np.where(comp=='Ne')[0][0]] = nt_Ne

nt_all = np.array([nt_Ar,nt_He,nt_H2,nt_CO,nt_CO2,nt_CH4,nt_H2O,nt_C2,nt_C3,nt_Ne,nt_C4,nt_C5,nt_C6])
ms_time_elap = np.vstack((ms_time_elap[:,7],ms_time_elap[:,1],ms_time_elap[:,0],ms_time_elap[:,6],ms_time_elap[:,9],ms_time_elap[:,2],ms_time_elap[:,3],ms_time_elap[:,5],ms_time_elap[:,8],ms_time_elap[:,4],ms_time_elap[:,-1],np.zeros(len(ms_time_elap[:,0])),np.zeros(len(ms_time_elap[:,0]))))
# Composition
x_i = nt_all/np.sum(nt_all,axis=0)

## Validation
# print('Validation:')
# for i in range(len(x_i_cal)):
#     print(comp[i],':',np.around(x_i[i]*sum(Vdot),1),'Real:',Vdot[i])


print('Composition calculated!')
messagebox.showinfo(message='Composition calculated!')



#===============================================================================
## Calculate the RSF values for ethane and propane
# Assumptions:
# - Gc measurements of ethane and propane are correct
# - He concentration previously calculated is used

gc_prompt = askquestion(title=None, message='Add GC calibration?')
if gc_prompt == 'yes':
    # Read the GC file (.npy)
    gc_filepath = filedialog.askopenfilename(initialdir=r'C:\Users\MaxGäßler\Documents\Rohdaten\GC',
                                            title='Select GC file for calibration',filetypes=[("Numpy files", ".npy")])
    # print(gc_filepath)
    # gc_filepath = r'C:/Users/simon/OneDrive - bwedu/Studium/02_Master/01_Masterarbeit/02_Versuche/03_Rohdaten/GC/036/SHA-E-036_GC_Daten.npy'
    gc_data = np.load(gc_filepath)
    gc_times = np.load(gc_filepath[:-12]+r'GC_times.npy')

    time_str = np.datetime_as_string(gc_times)
    ms_times_str = np.datetime_as_string(ms_times)

    try:
        np.where(ms_header_exp == 'Mass 30')[0][0]
        RSF_C2_30 = np.array([])
        mass26 = False
    except:
        np.where(ms_header_exp == 'C2')[0][0]
        RSF_C2_26 = np.array([])
        mass26 = True
    try:
        np.where(ms_header_exp == 'C3')[0][0]
        RSF_C3_43 = np.array([])
    except:
        np.where(ms_header_exp == 'Mass 44')[0][0]
        RSF_C3_44 = np.array([])

    RSF_C4_58 = np.array([])

    for t in range(1,len(time_str)):
        try:
            # print('\nTime:',time_str[t][11:19])
            # print(time_str[t][11:19])
            # search for the gc time in tn the ms time data and use the first appearance
            avg_idx = np.flatnonzero(np.core.defchararray.find(ms_times_str,time_str[t][11:19])!=-1)[0] 
            # print('Time in MS data:',ms_time_exp[avg_idx])
            # print(avg_idx)

            # average the ionic flows with 100 points
            ms_data_avg_gc = np.mean(ms_data_exp[avg_idx-50:avg_idx+50],axis=0) # average 100 points
            try:
                Idot_30 = ms_data_avg_gc[np.where(ms_header_exp == 'Mass 30')][0]
            except:
                Idot_26 = ms_data_avg_gc[np.where(ms_header_exp == 'C2')][0]
            try:
                Idot_43 = ms_data_avg_gc[np.where(ms_header_exp == 'C3')][0]
            except:
                Idot_44 = ms_data_avg_gc[np.where(ms_header_exp == 'Mass 44')][0]
            Idot_58 = ms_data_avg_gc[np.where(ms_header_exp == 'C4')][0]
            # Choose reference
            # print('\nHe as reference')
            x_ref = x_i[np.where(comp=='He')[0][0]-1,avg_idx] # x_ref = x_He as reference
            # print('Reference mole fraction:',x_ref)
            Idot_ref = ms_data_avg_gc[np.where(ms_header_exp=='He')[0][0]] # Idot_ref = Mass 4
            # print('Idot Helium:',Idot_ref)

            # Calculate RSF values
            # print('Mole fraction ethane:',gc_data[t,np.where(comp=='C2')[0,0]])
            # print('Mole fraction Propane:',gc_data[t,np.where(comp=='C3')[0,0]])
            # print('Ionic flow mass 30:',Idot_30)
            # print('Ionic flow mass 43:',Idot_43)
            # 
            # RSF_C2_26_temp = x_ref/gc_data[t,np.where(comp=='C2')[0][0]]*Idot_26/Idot_ref
            try:
                RSF_C2_30_temp = x_ref/gc_data[t,np.where(comp=='C2')[0][0]]*Idot_30/Idot_ref
            except:
                RSF_C2_26_temp = x_ref/(gc_data[t,np.where(comp=='C2')[0][0]]+gc_data[t,np.where(comp=='C2en')[0][0]])*Idot_26/Idot_ref # C2 = Ethan + Ethen
            try:
                # RSF_C3_43_temp = x_ref/gc_data[t,np.where(comp=='C3')[0][0]]*Idot_43/Idot_ref
                # C3 = Propan + Propen + Butane + Pentan + Hexan
                x_C3plus = gc_data[t,np.where(comp=='C3')[0][0]] + gc_data[t,np.where(comp=='C3en')[0][0]] + gc_data[t,np.where(comp=='C4')[0][0]] + gc_data[t,np.where(comp=='C5')[0][0]] + gc_data[t,np.where(comp=='C6')[0][0]]
                RSF_C3_43_temp = x_ref/x_C3plus*Idot_43/Idot_ref
            except:
                RSF_C3_44_temp = x_ref/gc_data[t,np.where(comp=='C3')[0][0]]*Idot_44/Idot_ref

            x_C4plus = gc_data[t,np.where(comp=='C4')[0][0]] + gc_data[t,np.where(comp=='C5')[0][0]] + gc_data[t,np.where(comp=='C6')[0][0]]
            RSF_C4_58_temp = x_ref/x_C4plus*Idot_58/Idot_ref
            
            # print('RSF C2:',RSF_C2_30_temp)
            # print('RSF C3:',RSF_C3_43_temp)

            try:
                RSF_C2_30 = np.append(RSF_C2_30, RSF_C2_30_temp)
            except:
                RSF_C2_26 = np.append(RSF_C2_26, RSF_C2_26_temp)
            try:
                RSF_C3_43 = np.append(RSF_C3_43, RSF_C3_43_temp)
            except:
                RSF_C3_44 = np.append(RSF_C3_44, RSF_C3_44_temp)
            
            RSF_C4_58 = np.append(RSF_C4_58, RSF_C4_58_temp)

        except:
            continue


    # print('\nFinished\n C2',np.around(RSF_C2_30,1))
    # print('C3:',np.around(RSF_C3_43,1))
    # print(RSF_C2_26.shape)
    # print(RSF_C3_43.shape)

    # Plot the RSF values
    try:
        plt.plot(np.arange(len(RSF_C2_30)),RSF_C2_30,label='Ethane, 30',marker='.')
    except:
        plt.plot(np.arange(len(RSF_C2_26)),RSF_C2_26,label='C2, 26',marker='.')
    try:
        plt.plot(np.arange(len(RSF_C3_43)),RSF_C3_43,label='C3plus, 43',marker='.')
    except:
        plt.plot(np.arange(len(RSF_C3_44)),RSF_C3_44,label='Propane, 44',marker='.')
    
    plt.plot(np.arange(len(RSF_C4_58)),RSF_C4_58,label='C4 plus, 58',marker='.')
    
    plt.ylabel('RSF / 1')
    plt.xlabel('GC timestep')
    plt.legend(frameon=False)
    plt.title(os.path.basename(path_exp)[:9])
    print(os.path.basename(path_exp[:9]))
    plt.show()

    gc_ind = askinteger(title='GC index',prompt='Choose index of GC measurement for calibration',initialvalue=3)

#=================================================================================================
### Final Concentration Calculation ### 

# Ethane
if gc_prompt == 'yes':
    try:
        print('RSF C2 Mass 30:',RSF_C2_30[gc_ind])
        nt_C2 = ms_data_exp[:,np.where(ms_header_exp=='Mass 30')[0][0]]/RSF_C2_30[gc_ind]
    except:
        print('RSF C2 Mass 26:',RSF_C2_26[gc_ind])
        if path_exp == r'C:/Users/simon/OneDrive - bwedu/Studium/02_Master/01_Masterarbeit/02_Versuche/03_Rohdaten/MS/SHA-E-064_260C_4_H2CO_20230204092754.txt': # by hand RSF for SHA-E-064
            print('064 erkannt')
            nt_C2 = ms_data_exp[:,np.where(ms_header_exp=='C2')[0][0]]/2 # RSF C2 Mass 26 = 2.0
        else:
            nt_C2 = ms_data_exp[:,np.where(ms_header_exp=='C2')[0][0]]/RSF_C2_26[gc_ind]  
else:
    # nt_C2 = ms_data_exp[:,np.where(ms_header_exp=='Mass 30')[0][0]]/RSF[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='Mass 30')[0][0]]
    nt_C2 = ms_data_exp[:,np.where(ms_header_exp=='C2')[0][0]]/RSF[np.where(comp=='C2')[0][0],np.where(ms_header_cal=='C2')[0][0]]
# Propane
if gc_prompt == 'yes':
    try:
        print('RSF C3 Mass 43:',RSF_C3_43[gc_ind])
        if path_exp == r'C:/Users/simon/OneDrive - bwedu/Studium/02_Master/01_Masterarbeit/02_Versuche/03_Rohdaten/MS/SHA-E-064_260C_4_H2CO_20230204092754.txt': # by hand RSF for SHA-E-064
            nt_C3 = ms_data_exp[:,np.where(ms_header_exp=='C3')[0][0]]/20 # RSF C3plus Mass 43 = 30
            print('064 erkannt')
        else:
            nt_C3 = ms_data_exp[:,np.where(ms_header_exp=='C3')[0][0]]/RSF_C3_43[gc_ind] 
    except:   
        print('RSF C3 Mass 44:',RSF_C3_44[gc_ind]) 
        nt_C3 = ms_data_exp[:,np.where(ms_header_exp=='Mass 44')[0][0]]/RSF_C3_44[gc_ind]
else:
    # nt_C3 = ms_data_exp[:,np.where(ms_header_exp=='Mass 43')[0][0]]/RSF[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='Mass 43')[0][0]]
    # nt_C3 = ms_data_exp[:,np.where(ms_header_exp=='Mass 44')[0][0]]/RSF[np.where(comp=='C3')[0][0],np.where(ms_header_cal=='Mass 44')[0][0]]
    nt_C3 = np.zeros_like(nt_Ar)
# C4 plus
if gc_prompt == 'yes':
    if path_exp == r'C:/Users/simon/OneDrive - bwedu/Studium/02_Master/01_Masterarbeit/02_Versuche/03_Rohdaten/MS/SHA-E-064_260C_4_H2CO_20230204092754.txt': # by hand RSF for SHA-E-064
        nt_C4 = ms_data_exp[:,np.where(ms_header_exp=='C4')[0][0]]/2 # RSF C4plus Mass 58 = 2
        print('064 erkannt')
    else:
        nt_C4 = ms_data_exp[:,np.where(ms_header_exp=='C4')[0][0]]/RSF_C4_58[gc_ind]
    
# All
nt_all = np.array([nt_Ar,nt_He,nt_H2,nt_CO,nt_CO2,nt_CH4,nt_H2O,nt_C2,nt_C3,nt_Ne,nt_C4,nt_C5,nt_C6])
# Composition
x_i = nt_all/np.sum(nt_all,axis=0)

# Test if H2O is stoichiometrically right
water_stoich = x_i[5,:] + x_i[7,:]*2 + x_i[8,:]*3

plt.figure()
plt.plot(ms_time_elap[6,:],x_i[6,:],label='Calc')
plt.plot(ms_time_elap[6,:],water_stoich[:],label='stoich')
plt.legend()


# Save the composition and time as numpy files
# storename = askstring('','Choose a filename\t\t\t\t',initialvalue=os.path.basename(path_exp[:-18]))
# # ms_path = Path(path_exp).parent.absolute()
# ms_path = r'C:\Users\simon\OneDrive - bwedu\Studium\02_Master\01_Masterarbeit\02_Versuche\04_Ausgewertet\MS'
# x_i_save = np.concatenate((np.array([ms_time_elap]),x_i),axis=0).T
# if gc_prompt == 'yes':
#     np.save(str(ms_path) + r'/' + storename + 'MS_times_with_GC.npy', ms_times)
#     np.save(str(ms_path) + r'/' + storename + 'MS_composition_with_GC.npy', x_i_save)
# else:
#     np.save(str(ms_path) + r'/' + storename + 'MS_times.npy', ms_times)
#     np.save(str(ms_path) + r'/' + storename + 'MS_composition.npy', x_i_save)



#---------------------------
# plotting the result
print('Plotting pre-results...')
fig,ax = plt.subplots()
for i in range(0,x_i.shape[0]):
    # ax.plot(np.arange(x_i.shape[1]),x_i[i,:]*sum(Vdot))
    # ax.plot(np.arange(x_i.shape[1]),x_i[i,:])
    ax.plot(ms_time_elap[i,:],x_i[i,:])
ax.set(xlabel='time',ylabel='composition / mol/mol',title=os.path.basename(path_exp)[:9])
# ax.legend(ms_header)
ax.legend(comp[1:])
ax.set_title('Preliminary results!')

fig,ax = plt.subplots()
ax.plot(ms_times,x_i[7,:],label='C2 MS')

if gc_prompt == 'yes':
    if mass26 == True:
        ax.plot(gc_times,gc_data[:,np.where(comp=='C2')[0][0]]+gc_data[:,np.where(comp=='C2en')[0][0]],label='C2 GC')
    else:
        ax.plot(gc_times,gc_data[:,np.where(comp=='C2')[0][0]]+gc_data[:,np.where(comp=='C2en')[0][0]],label='C2 GC')
    ax.set(xlabel='time',ylabel='composition / mol/mol',title=os.path.basename(path_exp)[:9])
    # ax.legend(ms_header)
    ax.legend(frameon=False)
    ax.set_title('Compare MS with GC results!')
    
if gc_prompt == 'yes':
    fig,ax = plt.subplots()
    ax.plot(ms_times,x_i[8,:],label='C3plus MS')
    x_C3plus_all = x_C3plus = gc_data[:,np.where(comp=='C3')[0][0]] + gc_data[:,np.where(comp=='C3en')[0][0]] + gc_data[:,np.where(comp=='C4')[0][0]] + gc_data[:,np.where(comp=='C5')[0][0]] + gc_data[:,np.where(comp=='C6')[0][0]]
    ax.plot(gc_times,x_C3plus_all,label='C3plus GC')
    ax.set(xlabel='time',ylabel='composition / mol/mol',title=os.path.basename(path_exp)[:9])
    # ax.legend(ms_header)
    ax.legend(frameon=False)
    ax.set_title('Compare MS with GC results!')

    fig,ax = plt.subplots()
    ax.plot(ms_times,x_i[10,:],label='C4plus MS')
    x_C4plus_all = gc_data[:,np.where(comp=='C4')[0][0]] + gc_data[:,np.where(comp=='C5')[0][0]] + gc_data[:,np.where(comp=='C6')[0][0]]
    ax.plot(gc_times,x_C4plus_all,label='C4plus GC')
    ax.set(xlabel='time',ylabel='composition / mol/mol',title=os.path.basename(path_exp)[:9])
    # ax.legend(ms_header)
    ax.legend(frameon=False)
    ax.set_title('Compare MS with GC results!')
    plt.show()
#=======================================================
ptk_prompt = askquestion(title=None, message='Cut the data for PTK analysis?')
if ptk_prompt == 'yes':
    # Choose start and end time
    starttime = askstring('Choose start time','Choose start time in the format:\n"DD.mm.YYYY HH:MM:SS.fff"',initialvalue=ms_time_exp[0])
    print('Start:',starttime)
    try:
        t_s_ptk = np.flatnonzero(np.core.defchararray.find(ms_time_exp,starttime)!=-1)[0] 
    except:
        messagebox.showerror(message='Chosen start time not part of the txt file!')
        raise ValueError('Chosen start time is not part of txt file!')
    endtime = askstring('Choose endtime','Choose end time in the format:\n"DD.mm.YYYY HH:MM:SS.fff"',initialvalue=ms_time_exp[-1])
    print('End:',endtime)
    try:
        t_e_ptk = np.flatnonzero(np.core.defchararray.find(ms_time_exp,endtime)!=-1)[0] 
    except:
        messagebox.showerror(message='Chosen end time is not part of the txt file!')
        raise ValueError('Chosen end time not part of txt file!')

    # Save the composition and time as numpy files
    storename = askstring('','Choose a filename\t\t\t\t',initialvalue=os.path.basename(path_exp[:-18]))
    # ms_path = Path(path_exp).parent.absolute()
    ms_path = r'Z:\cheming\zhk82\Projekte\FTS_dyn\AusgewerteteDateien'
    x_i_save = np.concatenate((np.array([ms_time_elap]),x_i),axis=0).T
    if gc_prompt == 'yes':
        np.save(str(ms_path) + r'/' + storename + 'MS_times_with_GC.npy', ms_times[t_s_ptk:t_e_ptk])
        np.save(str(ms_path) + r'/' + storename + 'MS_composition_with_GC.npy', x_i_save[t_s_ptk:t_e_ptk])
    else:
        np.save(str(ms_path) + r'/' + storename + 'MS_times.npy', ms_times[t_s_ptk:t_e_ptk])
        np.save(str(ms_path) + r'/' + storename + 'MS_composition.npy', x_i_save[t_s_ptk:t_e_ptk])
    #---------------------------
    # plotting the result
    print('Plotting...')
    fig,ax = plt.subplots()
    for i in range(1,x_i.shape[0]):
        ax.plot(ms_times[t_s_ptk:t_e_ptk],x_i_save[t_s_ptk:t_e_ptk,i])
    ax.set(xlabel='time',ylabel='composition / mol/mol',title=storename)
    # ax.legend(ms_header)
    ax.legend(comp[1:])
    plt.show()

else:
    # Save the composition and time as numpy files
    storename = askstring('','Choose a filename\t\t\t\t',initialvalue=os.path.basename(path_exp[:-18]))
    # ms_path = Path(path_exp).parent.absolute()
    ms_path = r'Z:\cheming\zhk82\Projekte\FTS_dyn\AusgewerteteDateien'
    #x_i_save = np.concatenate((np.array([ms_time_elap]),x_i),axis=0).T
    x_i_save = x_i.T
    time_save = ms_time_elap.T
    
    if gc_prompt == 'yes':
        np.save(str(ms_path) + r'/' + storename + 'MS_times_with_GC.npy', ms_time_elap)
        np.save(str(ms_path) + r'/' + storename + 'MS_composition_with_GC.npy', x_i_save)
    else:
        np.save(str(ms_path) + r'/' + storename + 'MS_times.npy', ms_times)
        np.save(str(ms_path) + r'/' + storename + 'MS_composition.npy', x_i_save)
    #---------------------------
    # plotting the result
    print('Plotting...')
    fig,ax = plt.subplots()
    for i in range(1,x_i.shape[0]):
        ax.plot(ms_times[:],x_i_save[:,i])
    ax.set(xlabel='time',ylabel='composition / mol/mol',title=storename)
    # ax.legend(ms_header)
    ax.legend(comp[1:])
    plt.show()


messagebox.showinfo(message='Data saved and script finished!')
#------------------------

print('Script finished!')