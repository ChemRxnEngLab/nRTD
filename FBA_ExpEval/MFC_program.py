'''
Script title: program_mfc_setpoints.py
Author: Franz Braun, Institute for Chemical Engineering, Ulm University --- Auf Basis von Vorarbeit von Simon Herrmann 
Description: Automate the programming of the MFC control of the helga test rig

Created on: 13.06.2025
'''


from pathlib import Path
import datetime
import shutil
import numpy as np


class MFCProgram:

    def __init__(self, template_path, mfc_number, store_path, store_name):
        # Initialize the script by setting the current working directory
        # Where is the template stored ? 
    
        self.template_path = template_path
        self.mfc_number = mfc_number
        self.store_name = store_name
        self.store_path = store_path
        # Ensure the store path directory exists
        Path(self.store_path).mkdir(parents=True, exist_ok=True)

    def create_new_script_Helga(self, file=None):
        return MFCProgram.create_new_script(self, file=file)

    def get_last_entry_number(self):
        with open(self.file_path,'r') as f:
            lines = f.readlines()
            last_idx = int(lines[-1][:3])
        return last_idx

    def set_line(self, index,time,setpoint,mfc_idx):
        '''
        Set the line in the script according to the format
        '''
        line_template = f'{index:03d}="{time:.1f};  {setpoint:.1f};  {mfc_idx:1d};  9"\n'
        line_template = line_template.replace(',','.')  # ('.',',') 
        return line_template


    def initialize_script(self, file):
        '''
        Sets all MFC setpoints to zero
        '''
        with open(file,'r') as f:
            lines = f.readlines()
        with open(file,'w') as f:
            ## delet all lines except the first 6
            f.writelines(lines[:6])
            ## Add a line for each MFC which sets it to zero
            lines2add = []
            for i in range(self.mfc_number):
                lines2add.append(MFCProgram.set_line(self, index=i+1,time=0,setpoint=0,mfc_idx=i+1))
            f.writelines(lines2add)

    def create_new_script(self, file=None):    
        '''
        Creates a new fps file with a script to control the mfcs and sets all mfcs to zero
        '''
        today = datetime.date.today().strftime("%Y%m%d")
        #Path(self.store_path + f'_{today}').mkdir(parents=True, exist_ok=True)    
        if file == None:
            now = datetime.datetime.now().strftime("%H%M%S")
            file = self.store_path + '\\' + self.store_name + f'_MFC_{today}.fps'
            print(f'Creating new script: {file}')
            self.file_path = file
        shutil.copyfile(self.template_path, file)
        MFCProgram.initialize_script(self, file)
        return file


    def add_line_to_file(self, file,time,setpoint,mfc_idx):
        '''
        Write the line to the file
        '''
        last_line_idx = MFCProgram.get_last_entry_number(file)
        with open(file,'a') as f:
            f.write(MFCProgram.set_line(self, last_line_idx+1,time,setpoint,mfc_idx))

    def add_mfc_change(self,time,setpoints):
        '''
        Adds lines for each mfc with the setpoint of 'setpoints' to the file.
        The number shape of setpoints needs to be the total number of mfcs
        '''
        last_line_idx = MFCProgram.get_last_entry_number(self)
        with open(self.file_path,'a') as f:
            for i in range(len(setpoints[:-1])):
                f.write(MFCProgram.set_line(self,last_line_idx+1+i,0,setpoints[i],mfc_idx=i+1))
            f.write(MFCProgram.set_line(self, last_line_idx+len(setpoints),time,setpoints[-1],len(setpoints)))

    def add_mfc_change_by_flowrates(self, duration, flowrates, COAr_or_Ar ):
        '''
        Adds lines for each mfc according to the flowrates given
        
        '''
        max_CO2_links = 100 # NmL/min
        max_H2_rechts = 250 # NmL/min
        max_H2_matrix = 1000 # NmL/min
        max_He_rechts = 200 # NmL/min
        if COAr_or_Ar== 'COAr':
            max_CO_rechts = 100 # NmL/min
        elif COAr_or_Ar == 'Ar':
            max_CO_rechts = 75 # NmL/min
        max_H2_links = 400 # NmL/min
        max_CO2_rechts = 75.05 # NmL/min
        max_He_links = 250 # NmL/min
        if COAr_or_Ar == 'COAr':
            max_CO_links = 100
        elif COAr_or_Ar == 'Ar':
            max_CO_links = 75
        max_Ne = 10 # NmL/min
        
        ## Calculate the setpoints for each MFC
        # Calculate the setpoints for each MFC
        setpoint_CO2_links = np.round(flowrates[0] / max_CO2_links * 100, 2)
        setpoint_H2_rechts = np.round(flowrates[1] / max_H2_rechts * 100, 2)
        setpoint_H2_matrix = np.round(flowrates[2] / max_H2_matrix * 100, 2)
        setpoint_He_rechts = np.round(flowrates[3] / max_He_rechts * 100, 2)
        setpoint_CO_rechts = np.round(flowrates[4] / max_CO_rechts * 100, 2)
        setpoint_H2_links = np.round(flowrates[5] / max_H2_links * 100, 2)
        setpoint_CO2_rechts = np.round(flowrates[6] / max_CO2_rechts * 100, 2)
        setpoint_He_links = np.round(flowrates[7] / max_He_links * 100, 2)
        setpoint_CO_links = np.round(flowrates[8] / max_CO_links * 100, 2)
        setpoint_Ne = np.round(flowrates[9] / max_Ne * 100, 2)
        setpoints = [setpoint_CO2_links, setpoint_H2_rechts, setpoint_H2_matrix, setpoint_He_rechts,
                    setpoint_CO_rechts, setpoint_H2_links, setpoint_CO2_rechts, setpoint_He_links,
                    setpoint_CO_links, setpoint_Ne]
        
        ## Add to file
        MFCProgram.add_mfc_change(self,
                    time=duration,
                    setpoints=setpoints)

