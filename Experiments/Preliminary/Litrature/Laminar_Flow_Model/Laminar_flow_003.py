import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import os

def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E = np.zeros_like(t)
    E[t >= tau / 2] = (tau**2) / (2 * (t[t >= tau / 2]**3))
    return E

t = np.linspace(0, 60, 500, endpoint=True)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

tau = 5.0

base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model'
E = laminarflow(t, tau)
print(f"tau: {tau}, integral E: {np.trapz(E, t)}")

c_out_full = np.convolve(c_0, E / np.sum(E), mode="full")
t_conv_full = np.linspace(t[0] + t[0], t[-1] + t[-1], len(c_out_full))
valid_indices = t_conv_full <= 60
t_conv = t_conv_full[valid_indices]
c_out = c_out_full[valid_indices]
tau_dir = os.path.join(base_dir, f'tau_500_{tau}')
os.makedirs(tau_dir, exist_ok=True)

np.save(os.path.join(tau_dir, 'time.npy'), t_conv)
np.save(os.path.join(tau_dir, 'concentration.npy'), c_out)


ax1.plot(t, E, label=f'tau {tau}')
ax2.plot(t_conv, c_out, label=f'tau {tau}')
ax1.set_ylim(0, 1.1 * np.max(E))
ax1.set_xlabel('t')
ax1.set_ylabel('E')
ax1.legend()

ax2.plot(t, c_0, label='c_0', linestyle='--', color='black')
ax2.set_ylim(0, 1.1)
ax2.set_xlim(0, 90)
ax2.set_xlabel('t')
ax2.set_ylabel('c')
ax2.legend()
#plt.savefig('laminar_flow_004_100disc.png', dpi=300)
plt.show()
t_conv = np.load(os.path.join(tau_dir, 'time.npy'))
c_out = np.load(os.path.join(tau_dir, 'concentration.npy'))

print(f"\nResults for tau = {tau:.2f}:")
print(f"Time: {t_conv[:10]}...")
print(f"Concentration: {c_out[:10]}...")
