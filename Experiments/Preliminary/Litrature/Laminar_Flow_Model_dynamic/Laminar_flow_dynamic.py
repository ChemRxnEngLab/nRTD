import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import os

noise_level = 0.001
tau = 5.0
base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model_dynamic'


def input_function(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 0.5 * (1.3 - np.cos(1.3 * t))
def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E = np.zeros_like(t)
    E[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E
t = np.linspace(0, 150, 300)  

input_func = input_function(t)
E = laminarflow(t, tau)
E_normalized = E / np.sum(E)
c_out_full = np.convolve(input_func, E_normalized, mode="full")
t_conv_full = np.linspace(t[0], t[0] + t[-1] * 2, len(c_out_full))
valid_indices = t_conv_full <= t[-1]
t_conv = t_conv_full[valid_indices]
c_out = c_out_full[valid_indices]

variance_scale = noise_level * np.var(c_out)
noise = np.random.normal(0, np.sqrt(variance_scale), c_out.shape)
c_out_noisy = c_out + noise

tau_dir = os.path.join(base_dir, f'tau_{tau}')
os.makedirs(tau_dir, exist_ok=True)
# np.save(os.path.join(tau_dir, 'time.npy'), t_conv)
# np.save(os.path.join(tau_dir, 'concentration.npy'), c_out_noisy)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
ax1.plot(t, E/E.max(), label=f'Laminar Flow E (tau = {tau})', color="red")
ax1.set_ylim(0, 1.1)
ax1.set_xlim(0, 50)
ax1.set_xlabel('Time')
ax1.set_ylabel('E')
ax1.legend()
ax2.plot(t_conv, c_out, label='Convolved Output c_out', color="blue")
ax2.plot(t_conv, c_out_noisy, label='Convolved Output_noisy c_out',linestyle='--', color="red",)
ax2.plot(t, input_func, label='Input Function', linestyle='--', color='black')
ax2.set_ylim(0, 1.1 )
ax2.set_xlim(0, 50)
ax2.set_xlabel('Time')
ax2.set_ylabel('Concentration')
ax2.legend()
plt.show()
