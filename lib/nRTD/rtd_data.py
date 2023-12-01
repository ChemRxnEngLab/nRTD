import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader


class RTDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # load the experimental data from files and save it in one tensor
        # called y containing n rows and m columns where n is the
        # number of samples and m is the number of time steps in the experimental data
        # y =

        # load the input signal from files and save it in one tensor
        # called X containing n rows and m columns where n is the
        # number of samples and m is the number of time steps in the input signal

        self.dataset = TensorDataset(X, y)

    def train_dataloader(self):
        # return a dataloader
        train_dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
