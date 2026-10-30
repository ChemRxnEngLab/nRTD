# Reads txt files with GC data and stores the data into a npy file into the same folder
# Exported text files from GC program need to be in ASCII format and have to include:
# - Data File Properties
# - Peak table
# - Identified Results
# - Grouping Result
# Not included are: Status Trace and Chromatogram

from tkinter import filedialog
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os





comp = np.array(['time','Ar','He','H2','CO','CO2','CH4','H2O','C2H6','C3H8','Ne','C4','C5','C6','C2en','C3en']) # Component names
indx =        [0,     1,    2,   3,   4,   5,    6,    7,    8,     9,     10,  11,  12,  13,  14,    15]

input_filenames = filedialog.askopenfilenames(initialdir=r'"C:\Users\FranzBraun\Desktop\PTK_FBA\Auswertung_FBA"',title='Select File')
# # input_filenames = [r'C:\Users\simon\OneDrive - bwedu\Studium\02_Master\01_Masterarbeit\02_Versuche\03_Rohdaten\GC\017\SHA-E-017_4bar_280C_4-1_0002.txt']

num = len(input_filenames)
# Replace comma with dot

for path in input_filenames:
    data = ""
    os.chmod(path,0o755)

    with open(path, 'r') as file:
        data = file.read().replace(',', '.')
        
    
    with open(path, "w") as out_file:
        out_file.write(data)

# Read the Concentration values and store them into npy file
data_gc = np.zeros((num,len(comp)))
gc_times = np.array([],dtype=np.datetime64)
for p in range(num):
    try:
        print('File:',input_filenames[p][-9:-4])
        # generate data frame from txt file
        DataFrame = pd.read_csv(input_filenames[p],sep='\t',header=73) # read the txt file with tab as delimiter and cut the first 73 filled lines
        data_array = np.array(DataFrame.values)
        # print(data_array.shape)
        # Evaluate the row index of each component in the data array
        ind_Ar = np.where(data_array[:,1] == 'Ar')[0][0]
        ind_CO = np.where(data_array[:,1] == 'CO')[0][0]
        ind_CH4 = np.where(data_array[:,1] == 'CH4')[0][0]
        ind_CO2 = np.where(data_array[:,1] == 'CO2')[0][0]
        ind_Ethan = np.where(data_array[:,1] == 'Ethan')[0][0]
        ind_Propan = np.where(data_array[:,1] == 'Propan')[0][0]
        if float(data_array[np.where(data_array[:,1] == 'n-Butane')[0][0],3]) != 0.: # check if n-butane is read correctly
            ind_Butan = np.where(data_array[:,1] == 'n-Butane')[0][0]
        else: # else the n-butane peak is read as trans-2-Butene
            ind_Butan = np.where(data_array[:,1] == 'trans-2-Butene')[0][0]
        ind_Pentan = np.where(data_array[:,1] == 'n-Pentane')[0][0]
        if float(data_array[np.where(data_array[:,1] == 'n-Hexane')[0][0],3]) != 0.: # check if n-hexane is read correctly
            ind_Hexan = np.where(data_array[:,1] == 'n-Hexane')[0][0]
        else: # else the n-hexane peak is read as C6-iso-8
            ind_Hexan = np.where(data_array[:,1] == 'C6-iso-8')[0][0]
        
        ind_Ethen = np.where(data_array[:,1] == 'Ethen')[0][0]
        ind_Propen = np.where(data_array[:,1] == 'Propen')[0][0]

        # Concentration of TCD detector
        # x_i in mol/mol
        data_gc[p,1] = float(data_array[ind_Ar,5])/100 # Ar
        data_gc[p,4] = float(data_array[ind_CO,5])/100 # CO
        data_gc[p,5] = float(data_array[ind_CO2,5])/100 # CO2
        data_gc[p,6] = float(data_array[ind_CH4,5])/100 # CH4
    
        # Concentration calculation by peak area of FID detector
        # Equation: x_i = A_i / (N_c_i * cal)
        # A_i ... peak area (3rd column in data array)
        # N_c_i ... number of c-atoms in molecule i
        # cal ... calibration factor
        cal = 212155057
        data_gc[p,8] = float(data_array[ind_Ethan,3])/2/cal   # C2
        data_gc[p,9] = float(data_array[ind_Propan,3])/3/cal  # C3
        data_gc[p,11] = float(data_array[ind_Butan,3])/4/cal  # C4
        data_gc[p,12] = float(data_array[ind_Pentan,3])/5/cal # C5
        data_gc[p,13] = float(data_array[ind_Hexan,3])/6/cal  # C6
        data_gc[p,14] = float(data_array[ind_Ethen,3])/6/cal  # C2 en
        data_gc[p,15] = float(data_array[ind_Propen,3])/6/cal  # C3 en

        # retrieve the elapsed time (time since first gc sample)
        DataFrame_time = pd.read_csv(input_filenames[p],sep='\t',header=13,nrows=1)
        data_array = np.array(DataFrame_time.values)
        time_str = data_array[0,1]
        date = datetime.strptime(time_str,'%d.%m.%Y %H:%M:%S')
        gc_times = np.append(gc_times,date)        
        # save start time
        if p == 0:
            start_time = date
        # calculate the elaped time
        elapsed_time = date - start_time
        # store in array
        data_gc[p,0] = elapsed_time.total_seconds()

        # # extract the measurement times
        # time_data = np.genfromtxt(input_filenames[p], delimiter='\t',skip_header=73,usecols=0,dtype=str)
        # print('deleting " ...')
        # short_time_data = np.char.replace(time_data, '"', '')
        # print('Converting to datetime ...')
        # pd_times = pd.to_datetime(short_time_data,format='%d.%m.%Y %H:%M:%S.%f')
        # print('Creating final array...')
        # np_times = np.array(pd_times,dtype=np.datetime64)

    except: # if FID detects no components
        print('File:',input_filenames[p][-9:-4],'(No FID data)')
        # generate data frame from txt file
        DataFrame = pd.read_csv(input_filenames[p],sep='\t',header=75) 
        data_array = np.array(DataFrame.values)
        # print(data_array)
        # Evaluate the row index of each component in the data array
        ind_Ar = np.where(data_array[:,1] == 'Ar')[0][0]
        ind_CO = np.where(data_array[:,1] == 'CO')[0][0]
        ind_CH4 = np.where(data_array[:,1] == 'CH4')[0][0]
        ind_CO2 = np.where(data_array[:,1] == 'CO2')[0][0]
        
        # Concentration of TCD detector
        # x_i in mol/mol
        data_gc[p,1] = float(data_array[ind_Ar,5])/100 # Ar
        data_gc[p,4] = float(data_array[ind_CO,5])/100 # CO
        data_gc[p,5] = float(data_array[ind_CO2,5])/100 # CO2
        data_gc[p,6] = float(data_array[ind_CH4,5])/100 # CH4

        # retrieve the elapsed time (time since first gc sample)
        DataFrame_time = pd.read_csv(input_filenames[p],sep='\t',header=13,nrows=1) # read the txt file with tab as delimiter and cut the first 13 filled lines and only evaluate 1 row
        data_array = np.array(DataFrame_time.values)
        time_str = data_array[0,1]
    
        date = datetime.strptime(time_str,'%d.%m.%Y %H:%M:%S')
        # Append datetime to times array
        gc_times = np.append(gc_times,date) 
        # save start time
        if p == 0:
            start_time = date
        # calculate the elaped time
        elapsed_time = date - start_time
        # store in array
        data_gc[p,0] = elapsed_time.total_seconds()

# print(data_gc)

# Correct the datetime of the GC PC towards the MS PC
# Time difference is 18 minutes and 10 seconds
corr_time_min = np.timedelta64(18,'m')
corr_time_sec = np.timedelta64(10,'s')
gc_times = gc_times.astype(np.datetime64)+corr_time_min+corr_time_sec

# save as .npy file
gc_data_path = input_filenames[0][:-9]
np.save(gc_data_path+'_GC_Daten.npy',data_gc)
np.save(gc_data_path+'_GC_times.npy',gc_times.astype(np.datetime64))
print('Saved as:', gc_data_path+'_GC_Daten.npy')

# Plot the profiles
fig,ax = plt.subplots()
## with elapsed time since start
# ax.plot(data_gc[:,0]/3600,data_gc[:,1:],marker='.') 
# ax.set_xlabel('time / h')
## with measurement time
ax.plot(gc_times,data_gc[:,1:],marker='.')
labels = comp[1:]
#ax.legend(labels=labels)
ax.set_ylabel('x_i / mol/mol')
ax.set_xlabel('time')
plt.show()


