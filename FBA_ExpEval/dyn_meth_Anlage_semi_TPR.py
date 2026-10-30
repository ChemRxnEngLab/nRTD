'''
The class semi TPR enables the evaluation of the reduction in the dyn-meth test rig
therefore the Jumo data and the MS - H2O signal are plotted against each other 
Warning - this skript does not convert any signals in molar / mass flows --- the unit is arbitrary
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ICIWstyle")
from ICIW_Plots import cm2inch as cm2inch
from ICIW_Plots import make_square_ax
from ICIW_Plots import make_square_subplots
#from ICIW_Plots.figures import Elsevier_Sizes, ACS_Sizes
from datetime import datetime, timedelta

class semiTPR:
    def __init__(self, MS_data_path, Jumo_data_path, comp_MS, year, day_corrector=0, month_corrector=0,day_per_month = 31, time_of_tpr = 16*3600):
        self.MS_data_path = MS_data_path            # Path to the MS data
        self.Jumo_data_path = Jumo_data_path        # Path to the Jumo data
        self.comp_MS = comp_MS                      # Components in the MS data
        self.year = year                            # Year of the experiment
        self.day_corrector = day_corrector          # Day corrector for the Jumo data
        self.month_corrector = month_corrector      # Month corrector for the Jumo data
        self.day_per_month = day_per_month          # if two moths - number of days of first month 
        self.time_of_tpr = time_of_tpr              # Time of TPR in seconds 
    
        df = np.loadtxt(Jumo_data_path, delimiter="\t", max_rows=2, usecols=(1,2,3,4,5), dtype=str,skiprows=6)
        self.header_jumo_Co_TiO2 = df
        df = np.loadtxt(Jumo_data_path, delimiter="\t", usecols=(1,2,3,4,5), dtype=str,skiprows=8)
        self.Jumo_times = pd.to_datetime(df[:,1],format='"%H:%M:%S"')
        self.t_start_Jumo_TiO2 = self.Jumo_times[0]

        Jumo_hours_min_sec = pd.to_datetime(df[:,1],format='"%H:%M:%S"')
        times_days_month_years = pd.to_datetime(df[:,0], format='"%d.%m.%Y"')

        
        


        self.Jumo_seconds = Jumo_hours_min_sec.hour * 3600 + Jumo_hours_min_sec.minute * 60 + Jumo_hours_min_sec.second + times_days_month_years.day * 24 * 3600 + times_days_month_years.month * self.day_per_month * 24 * 3600 + self.day_corrector * 24 * 3600 + self.month_corrector * 30 * 24 * 3600 
        self.Jumo_time_delta = np.array((self.Jumo_seconds - self.Jumo_seconds[0])) #self.Jumo_seconds - self.Jumo_seconds[0]    
        #print(Jumo_seconds - Jumo_seconds[0])
        # Convert Jumo_seconds[0] to a datetime object
        self.start_year= datetime(year, 1, 1)  # Unix epoch start
        self.Jumo_start = self.start_year + timedelta(seconds=int(self.Jumo_seconds[0]))
        print(self.Jumo_start)
   
        
        self.data_Co_TiO2 = np.array((df))

        
        self.data_Co_TiO2[:,0]  = np.array((self.Jumo_time_delta))  
        for i in np.arange(0,len(self.data_Co_TiO2[:,0])):
            for j in np.arange(0,len(self.data_Co_TiO2[0,:])):
                self.data_Co_TiO2[i,j] = self.data_Co_TiO2[i,j].replace('"','')
        self.data_Jumo_Co_TiO2 = np.array((self.data_Co_TiO2[:,2:])).astype(float)

        plt.plot(self.Jumo_time_delta, self.data_Jumo_Co_TiO2[:,2])
        plt.xlabel('Jumo Time Delta (s)')
        plt.ylabel('T_TiO2')
        plt.title('T_TiO2 over Jumo Time Delta')
        plt.show()


        'make each entry in species a variable and assign the corresponding index to it as value (e.g. Ar = 1, He = 2, H2 = 3, ...)'
        'give each species a specific color'
        self.color = []
        for cc in self.comp_MS:
            exec(f"self." + f"{cc}  = {np.where(self.comp_MS==cc)[0][0]} ")
            self.color.append(f"C" + f"{np.where(self.comp_MS==cc)[0][0]}") 

        self.ms_exp           = pd.read_csv(self.MS_data_path,delimiter='\t',skiprows=2)
        self.ms_times         = pd.to_datetime(self.ms_exp['Time'],format='%d.%m.%Y %H:%M:%S.%f')
        self.t_start_MS_TiO2 = self.ms_times[0]
        self.ms_time_delta    = []
        self.ms_times_Jumo_delta = []
        for s in self.ms_times:
            if self.ms_times[0] > s:
                s += pd.Timedelta(hours=24)
            self.ms_time_delta.append((s - self.ms_times[0]).total_seconds())
        self.ms_time_delta  = np.array((self.ms_time_delta))
        self.ms_times_Jumo_delta  = np.array((self.ms_times_Jumo_delta))
        self.ms_H2O = self.ms_exp['Mass 18']

        plt.plot(self.ms_time_delta, self.ms_H2O)
        plt.xlabel('MS Time Delta (s)')
        plt.ylabel('MS H2O Signal')
        plt.title('MS H2O Signal over MS Time Delta')
        plt.show()


        print(f"Jumo Start Time: {self.Jumo_start}")
        print(f"MS Start Time: {self.t_start_MS_TiO2}")

        # Calculate the time difference
        self.time_difference = (self.t_start_MS_TiO2 - self.Jumo_start).total_seconds()
        # Extract minutes, seconds, and milliseconds from the time difference
        # self.time_difference_minutes = self.time_difference.components.minutes
        # self.time_difference_seconds = self.time_difference.components.seconds
        # self.time_difference_milliseconds = self.time_difference.components.milliseconds

        # print(f"Minutes: {self.time_difference_minutes}, Seconds: {self.time_difference_seconds}, Milliseconds: {self.time_difference_milliseconds}")
        # # Calculate the total time difference in seconds
        # total_time_difference_seconds = (self.time_difference_minutes * 60) + self.time_difference_seconds + (self.time_difference_milliseconds / 1000)
        # print(f"Total Time Difference in Seconds: {total_time_difference_seconds}")
        print(f"Time Difference in Seconds: {self.time_difference}")

        # Define the number of interpolation steps
        num_steps = 10000

        print(self.ms_time_delta[-1])
        print(self.Jumo_time_delta[-1])
        # Interpolate Jumo values
        # Adjust the start and end times for Jumo interpolation
        self.index_start_Jumo = np.where(self.Jumo_time_delta >= self.time_difference)[0][0]
        # if self.Jumo_time_delta[-1] > self.ms_time_delta[-1]:
        #     self.index_end_Jumo   = np.where(self.Jumo_time_delta >= self.ms_time_delta[-1]+self.time_difference)[0][0]
        #     self.index_end_MS     = self.ms_time_delta.size
        # else:
        #     self.index_end_Jumo   = self.Jumo_time_delta.size
        #     self.index_end_MS     = np.where(self.ms_time_delta+self.time_difference >= self.Jumo_time_delta[-1])[0][0]
        time_of_tpr
        self.index_end_Jumo   = np.where(self.Jumo_time_delta >= self.time_of_tpr+self.time_difference)[0][0]
        self.index_end_MS     = np.where(self.ms_time_delta >= self.time_of_tpr)[0][0]
        # Generate the interpolation time array within the specified range
        self.interp_time = np.linspace(0, 18*3600, num_steps)

        # Interpolate Jumo values within the adjusted time range
        self.jumo_interp_values = np.interp(self.interp_time + self.time_difference, self.Jumo_time_delta[self.index_start_Jumo:self.index_end_Jumo],self.data_Jumo_Co_TiO2[self.index_start_Jumo:self.index_end_Jumo,2])

        # interpolate ms values
        self.ms_interp_values = np.interp(self.interp_time, self.ms_time_delta[:self.index_end_MS] ,self.ms_H2O[:self.index_end_MS])


        print(self.ms_interp_values.size)
        print(self.jumo_interp_values.size)

        plt.plot(self.jumo_interp_values[:-2000],self.ms_interp_values[:-2000])
        #plt.title('Co-TiO2')
        plt.xlabel(r'$T\,/ \,\mathrm{°C}$')
        plt.ylabel(r'MS H2O Signal')
        plt.show()


    

