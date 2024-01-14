# import pytorch_lightning as pl
# from torch.utils.data import TensorDataset, DataLoader
# import numpy as np
# import torch

# class RTDDataModule(pl.LightningDataModule):
#     def __init__(self, batch_size: int = 4):
#         super().__init__()
#         self.batch_size = batch_size

#     def prepare_data(self):

#         # You may need to define 'y' based on your specific data structure.
#         # For example, if 'y' is another NumPy file, load it in a similar manner.
#         # y_numpy = np.load('/path/to/y.npy')
#         # y = torch.from_numpy(y_numpy)

#         # Assuming 'y' is defined or loaded from a file, create the TensorDataset
#         # Note: Make sure that x, t, and y have compatible shapes
#         self.dataset = TensorDataset(x, t)

import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import glob

class RTDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4, data_folder: str = '/path/to/data_folder'):
        super().__init__()
        self.batch_size = batch_size
        self.data_folder = data_folder

    def prepare_data(self):
        c_out = sorted(glob.glob('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1_TOA_MGA_20231020_009_000001_x_*.npy'))
        t_conv = sorted(glob.glob('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1_TOA_MGA_20231020_009_000001_t_*.npy'))
        target_file_x = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/Data_Scripts_Evaluation/Experiment_1/H_085_C1_S_009/x_target.npy'
        target_file_t = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/Data_Scripts_Evaluation/Experiment_1/H_085_C1_S_009/t_target.npy'

        x_tensors = [torch.from_numpy(np.load(file)) for file in c_out]
        t_tensors = [torch.from_numpy(np.load(file)) for file in t_conv]
        target_tensors_x = torch.from_numpy(np.load(target_file_x))
        target_tensors_t = torch.from_numpy(np.load(target_file_t))

        # Assuming 'y' is defined or loaded from files, create the TensorDataset
        # Note: Make sure that x, t, and y have compatible shapes
        self.dataset = TensorDataset(*x_tensors, *t_tensors, target_tensors_x, target_tensors_t)

    def train_dataloader(self):
        # Return a DataLoader
        train_dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return train_dl

