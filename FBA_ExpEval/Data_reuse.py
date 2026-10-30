'''
These functions allow saving objects generated in other modules and loading these in any project using pickle.
suggested workflow: 
one file to evaluate the raw data (lab journal), save the relevant objects using the save function,
then load these objects in another file to reuse them for further evaluations or plotting.
'''

import numpy as np
import json
import sys
import os
import pickle as pkl
from datetime import datetime


def save( obj, storepath, storename):
        """
        Save an object to a file using pickle.
        """

        current_date = datetime.now().strftime('%Y%m%d')
        file_name = f"{storename}_{current_date}.pkl"
        results_dir = os.path.join(storepath, 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Create the full file path inside the directory
        global file_path
        file_path = os.path.join(results_dir, file_name)  # or use f"{storename}.pkl"
        print(file_path)
        with open(file_path, 'wb') as f:
            pkl.dump(obj, f)


    
def load (file_path):
        """        
        Load an object from a file using pickle.
        """

       # Load the object
        with open(file_path, 'rb') as f:
            loaded_obj = pkl.load(f)

        return loaded_obj
    