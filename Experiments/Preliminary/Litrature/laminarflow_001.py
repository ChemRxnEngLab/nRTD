import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import os

def laminarflow(t: npt.NDArray[np.float64], tau: float) -> npt.NDArray[np.float64]:
    E=np.zeros_like(t)
    E[t>=tau/2]=(tau**2) / (2 * (t[t>=tau/2]**3))
    return E

t = np.linspace(0, 30, 500, endpoint=True)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
tau_values = np.linspace(1, 5, 2)

base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model'


for tau in tau_values:
    E = laminarflow(t, tau)
    print(f"tau: {tau}, integral E: {np.trapz(E, t)}")
    c_out = np.convolve(c_0, E / np.sum(E), mode="full")
    t_conv = np.linspace(t[0] + t[0], t[-1] + t[-1], len(c_out))
    
    tau_dir = os.path.join(base_dir, f'tau_{int(tau)}')
    os.makedirs(tau_dir, exist_ok=True)
    
    np.save(os.path.join(tau_dir, 'time.npy'), t_conv)
    np.save(os.path.join(tau_dir, 'concentration.npy'), c_out)
   
    ax1.plot(t, E, label=f'tau {tau:.4f}')
    ax2.plot(
        t_conv,
        c_out,
        label=f'tau {tau:.4f}')
ax1.set_ylim(0, 1.1 * np.max([np.max(laminarflow(t, g)) for g in tau_values]))
ax1.set_xlabel('t')
ax1.set_ylabel('E')
ax1.legend()
ax2.plot(t, c_0, label='C_0', linestyle='--', color='black')
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('t')
ax2.set_ylabel('C')
ax2.legend()
plt.savefig('laminar_flow.png', dpi=300)
plt.show()

for tau in tau_values:
    tau_dir = os.path.join(base_dir, f'tau_{int(tau)}')
    t_conv = np.load(os.path.join(tau_dir, 'time.npy'))
    c_out = np.load(os.path.join(tau_dir, 'concentration.npy'))
    print(f"\nResults for tau = {tau}:")
    print(f"Time: {t_conv[:10]}...")  # Print first 10 time points
    print(f"Concentration: {c_out[:10]}...")  # Print first 10 concentration values
