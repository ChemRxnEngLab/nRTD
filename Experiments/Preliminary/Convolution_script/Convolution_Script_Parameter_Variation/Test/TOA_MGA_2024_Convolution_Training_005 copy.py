import sys

sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")
#sys.path.append("lib")
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import matplotlib.pyplot as plt
from nrtd import RTDModule
import numpy as np
import wandb


n_disc = 377
t_input = torch.linspace(0, 31, n_disc)
c_in = torch.zeros((20, 1, n_disc))
c_in[::2, :, t_input > 1] = 0.05
c_in[1::2, :, t_input < 1] = 0.05
file_numbers = range(1, 21)
c_out_list = []
t_conv_list = []
#file_numbers = range(1, 21, 2)

for i, file_num in enumerate(file_numbers):
    t_conv_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_t_processed.npy"
   # t_conv_path = f"Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_t_processed.npy"
    c_out_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_x_processed.npy"
   # c_out_path = f"Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_x_processed.npy"
    
   #print(f"Processing files: {t_conv_path}, {c_out_path}")
   
    t_conv = (torch.tensor(np.load(t_conv_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    c_out = (torch.tensor(np.load(c_out_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    c_out_list.append(c_out)
    t_conv_list.append(t_conv)
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)
plt.show()
print(f"c_in size: {c_in.size()}")
print(f"c_out size: {c_out.size()}")
print(f"t_conv size: {t_conv.size()}")

pl.seed_everything(42, workers=True)#check tuana

model = RTDModule(
    kernel_size=122,
    learning_rate=1e-1,
)
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)
j = 2

plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label=f"Exp_{file_numbers[i]}",
        color="green",
    )

plt.plot(
    t_conv[j, 0, :].numpy(),
    c_conv[j, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.legend()
plt.xlim((0, 10))
plt.ylim((0, 0.2))
plt.show()

print(model(c_in).size())

ds = TensorDataset(c_in, c_out)
# split the dataset into train and test 
#train_ds, test_ds = random_split(ds, [0.8, 0.2])

# train_dl = DataLoader(train_ds, batch_size=1)
# test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

dl = DataLoader(ds, batch_size=20, shuffle=False)

# set up the logger
wandb_logger = pl_loggers.WandbLogger(
    project="nRTD",
    log_model=True,
)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=1000,  
    logger=wandb_logger,
    deterministic=True#tuana checkkk
)


#trainer.fit(model, train_dl)
trainer.fit(model, dl)
# adds an epoch at the end to calculate the final loss at the traineing end
#trainer.test(model, test_dl)
trainer.test(model, dl)### check it maybe you will see changes??
wandb.finish()

##################
# Postprocessing #
##################

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

# Iterate over the loaded files and plot them
for i in range(c_out.size(1)):
    plt.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label=f"Exp_{file_numbers[i]}",
        color="green",
    )

plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.xlim((0, 10))
plt.ylim((0, 0.2))
plt.savefig("Figure_005_010_False")
plt.legend()
plt.show()


