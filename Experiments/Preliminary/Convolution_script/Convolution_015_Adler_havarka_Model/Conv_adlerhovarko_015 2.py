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
import sympy as sp
import wandb
from nrtd import RTDModule

tau_5_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Adler_havarka_Model/tau_a_val_1_tau_p_val_2_tau_m_val_0.4000000000000001_beta_val_0.1'

t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)


n_disc = 277
t_input = torch.linspace(0, 20, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 1] = 1
c_in[1::2, :, t_input < 1] = 1

#file_numbers = range(1, 21)
c_out_list = []
t_conv_list = []

c_out_list.append(c_out_tau)
t_conv_list.append(t_conv_tau)


# for i, file_num in enumerate(file_numbers):

#     pass


c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

plt.show()
print(f"c_in size: {c_in.size()}")
print(f"c_out size: {c_out.size()}")
print(f"t_conv size: {t_conv.size()}")

model = RTDModule(
    kernel_size=222,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 25, model.kernel_size)
E = E / E.max()
j = 0  

plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

# for i in range(c_out.size(1)):
#     plt.plot(
#         t_conv[j, i, :].numpy(),
#         c_out[j, i, :].numpy(),
#         label=f"Exp_{file_numbers[i]}" if i < len(file_numbers) else "Tau 1",
#         color="green",
#     )
for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label= "Tau 5.0",
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
    max_epochs=10000,
    deterministic=True
)
trainer.fit(model, dl)
trainer.test(model, dl)


c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 25, model.kernel_size)
E = E / E.max()

def compute_inverse_laplace(coefficients, t_values):
    s, t = sp.symbols('s t', real=True, positive=True)
    alpha = coefficients['alpha_val']
    results = []  
    for tau_a_val in coefficients['tau_a_val']:
        for tau_p_val in coefficients['tau_p_val']:
            for beta_val in coefficients['beta_val']:
                tau_m_val = (beta_val * (1 - alpha)) / alpha
                # Laplace transform equation
                F_s = (sp.exp(-tau_p_val * s)) / (1 + beta_val + tau_a_val * s - (beta_val / (1 + tau_m_val * s)))
                f_t = sp.inverse_laplace_transform(F_s, s, t)
                f_t_numeric = sp.lambdify(t, f_t, modules="numpy")  # Convert to numerical function
                E_t = f_t_numeric(t_values)
                results.append({
                    'tau_a_val': tau_a_val,
                    'tau_p_val': tau_p_val,
                    'tau_m_val': tau_m_val,
                    'beta_val': beta_val,
                    'E_t': E_t
                })
    return results
coefficients = {
    'tau_a_val': np.array([1]),
    'tau_p_val': np.array([2]),
    'beta_val': np.array([0.1]),
    'alpha_val': 0.2
}
t_plot = torch.linspace(7, 40, 500).numpy()
inverse_laplace_results = compute_inverse_laplace(coefficients, t_plot)
E_expected = inverse_laplace_results[0]['E_t']



plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="Bo10",
        color="green",
    )

plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.plot(t_plot, E_expected, label="E_predicted", color="purple")
plt.xlim((0, 50))
plt.ylim((0, 1.1))
plt.legend()

#plt.savefig("Figure_Adler_havarka_Model_001")
plt.show()
