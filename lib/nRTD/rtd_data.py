import lightning.pytorch as pl


class RTDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str = None):
        # assign train/val datasets for use in dataloaders
        # called on every GPU
        pass

    def train_dataloader(self):
        # return a dataloader
        pass

    def val_dataloader(self):
        # return a dataloader
        pass

    def test_dataloader(self):
        # return a dataloader
        pass
