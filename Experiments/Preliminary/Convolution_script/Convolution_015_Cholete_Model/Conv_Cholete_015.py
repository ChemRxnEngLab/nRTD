import sys
import os
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import matplotlib.pyplot as plt
import numpy as np
import wandb
from nrtd import RTDModule

# Bo_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Cholete_Model/beta0.9'

# t_conv_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# c_out_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# n_disc = 119
# t_input = torch.linspace(0, 40, n_disc)
# c_in = torch.zeros((1, 1, n_disc))
# c_in[::2, :, t_input > 5] = 1
# c_in[1::2, :, t_input < 5] = 1

# #file_numbers = range(1, 21)
# c_out_list = []
# t_conv_list = []

# c_out_list.append(c_out_tau)
# t_conv_list.append(t_conv_tau)


# # for i, file_num in enumerate(file_numbers):

# #     pass


# c_out = torch.cat(c_out_list, dim=0)
# t_conv = torch.cat(t_conv_list, dim=0)

# plt.show()
# print(f"c_in size: {c_in.size()}")
# print(f"c_out size: {c_out.size()}")
# print(f"t_conv size: {t_conv.size()}")

# model = RTDModule(
#     kernel_size=80,
#     learning_rate=10e-3,
#     use_scheduler=True,
#     scheduler_kwargs={"factor": 0.5, "patience": 80},
# )
# c_conv = model(c_in)
# E = model.net.E[0]
# t_E = torch.linspace(5, 60, model.kernel_size)
# E = E / E.max()
# j = 0  

# plt.figure()
# plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

# # for i in range(c_out.size(1)):
# #     plt.plot(
# #         t_conv[j, i, :].numpy(),
# #         c_out[j, i, :].numpy(),
# #         label=f"Exp_{file_numbers[i]}" if i < len(file_numbers) else "Tau 1",
# #         color="green",
# #     )
# for i in range(c_out.size(1)):
#     plt.plot(
#         t_conv[j, i, :].numpy(),
#         c_out[j, i, :].numpy(),
#         label= "Tau 5.0",
#         color="green")
# plt.plot(
#     t_conv[j, 0, :].numpy(),
#     c_conv[j, 0, :].detach().numpy(),
#     label="Predicted",
#     color="red",
# )
# plt.plot(t_E, E, label="E", color="orange")
# plt.legend()
# plt.xlim((0, 40))
# plt.ylim((0,1.1))
# plt.show()

# print(model(c_in).size())


# ds = TensorDataset(c_in, c_out)
# dl = DataLoader(ds, batch_size=20, shuffle=True)

# # wandb_logger = pl_loggers.WandbLogger(
# #     project="nRTD",
# #     log_model=True)

# # trainer = pl.Trainer(
# #     accelerator="auto",
# #     max_epochs=10000,
# #     logger=wandb_logger, deterministic=True
# # )

# trainer = pl.Trainer(
#     accelerator="auto",
#     max_epochs=10000, deterministic=True
# )

# trainer.fit(model, dl)
# trainer.test(model, dl)


# c_conv = model(c_in)
# E = model.net.E[0]
# t_E = torch.linspace(5, 60, model.kernel_size)
# E = E / E.max()

# def Cholete(t: npt.NDArray[np.float64], alpha: float, beta: float, tau: float, g: float) -> npt.NDArray[np.float64]: 
#     print(f"tau = {tau}")
#     print(f"alpha = {alpha}")
#     print(f"beta = {beta}")
#     print(f"g = {g}")
    
#     H = np.where(t < g, 0, 1)
#     k = ((1 - alpha) / (beta * tau))
#     exp_term = (1 - alpha) * np.exp(k * (g - t))
#     F = alpha * H - exp_term + (1 - alpha)
#     F[F < 0] = 0
#     E_expected = np.gradient(F,t)  
#     return F,E_expected


# plt.figure()
# plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

# for i in range(c_out.size(1)):
#     plt.plot(
#         t_conv[0, i, :].numpy(),
#         c_out[0, i, :].numpy(),
#         label="beta0.9",
#         color="green",
#     )

# plt.plot(
#     t_conv[0, 0, :].numpy(),
#     c_conv[0, 0, :].detach().numpy(),
#     label="Predicted",
#     color="red",
# )
# plt.plot(t_E, E, label="E", color="orange")
# plt.xlim((0, 40))
# plt.ylim((0, 1.1))
# plt.legend()

# # fig = plt.gcf()
# # wandb.log({"RTD_Plot": fig})
# # wandb.log({"RTD_Plot": wandb.Image(fig)})


# #wandb.finish()




import sys
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import matplotlib.pyplot as plt
import numpy as np
import wandb
from nrtd import RTDModule
import numpy.typing as npt

module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)

Bo_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Cholete_Model/beta0.3'
t_conv_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(Bo_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

n_disc = 119
t_input = torch.linspace(0, 40, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 5] = 1
c_in[1::2, :, t_input < 5] = 1

c_out_list = []
t_conv_list = []
c_out_list.append(c_out_tau)
t_conv_list.append(t_conv_tau)

c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

model = RTDModule(
    kernel_size=80,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 60, model.kernel_size)
E = E / E.max()
j = 0  

plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label= "beta0.9",
        color="green")
plt.plot(
    t_conv[j, 0, :].numpy(),
    c_conv[j, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.legend()
plt.xlim((0, 40))
plt.ylim((0,1.1))
plt.show()

print(model(c_in).size())

ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000, deterministic=True
)

trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 60, model.kernel_size)
# E = E / E.max()

def Cholete(t: npt.NDArray[np.float64], alpha: float, beta: float, tau: float, g: float) -> npt.NDArray[np.float64]: 
    shifted_t = t - 5  
    shifted_t[shifted_t < 0] = 0  
    H = np.where(shifted_t < g, 0, 1)
    k = ((1 - alpha) / (beta * tau))
    exp_term = (1 - alpha) * np.exp(k * (g - shifted_t))
    F = alpha * H - exp_term + (1 - alpha)
    F[F < 0] = 0  
    E_expected = np.gradient(F, t)  
    E_expected = E_expected / E_expected.max()  
    return F, E_expected

alpha = 0.5
beta = 0.3
tau = 5.0
g = 5.0
t_values = torch.linspace(0, 60, 200).numpy()

F, E_expected = Cholete(t_values, alpha, beta, tau, g)

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="beta0.9",
        color="green",
    )

plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_values, E_expected, label="E (Expected)", color="purple", linestyle="--")
plt.plot(t_E+10, E, label="E", color="orange")
plt.xlim((0, 70))
plt.ylim((0, 1.1))
plt.legend()
plt.savefig("Figure_conv_Cholete_beta03")
plt.show()

