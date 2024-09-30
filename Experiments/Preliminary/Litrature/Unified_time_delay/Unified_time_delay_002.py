import numpy as np
import matplotlib.pyplot as plt
import os

def laminarflow(t: np.ndarray, J: int, tau: int, alpha: float) -> np.ndarray:
    if J == 2 and tau == 1 and alpha == 0.2:
        E_t = (3.24642 * np.exp(-2.34833 * t) * t
               + 0.0392897 * np.exp(-0.851669 * t) * t
               - 0.477252 * np.exp(-2.34833 * t)
               + 0.477252 * np.exp(-0.851669 * t))
    elif J == 5 and tau == 1 and alpha == 0.2:
        E_t = (123.224 * np.exp(-5.24709 * t) * t**4
               + 5.66367e-9 * np.exp(-0.952909 * t) * t**4
               - 6.36342 * np.exp(-5.24709 * t) * t**3
               + 8.68252e-6 * np.exp(-0.952909 * t) * t**3
               - 4.34703 * np.exp(-5.24709 * t) * t**2
               + 0.00108703 * np.exp(-0.952909 * t) * t**2
               - 1.97921 * np.exp(-5.24709 * t) * t
               + 0.0449 * np.exp(-0.952909 * t) * t
               - 0.450448 * np.exp(-5.24709 * t)
               + 0.450448 * np.exp(-0.952909 * t))
    elif J == 5 and tau == 3 and alpha == 0.2:
        E_t = (0.507094 * np.exp(-1.74903 * t) * t**4
               + 9.81543e-11 * np.exp(-0.317636 * t) * t**4
               - 0.0785608 * np.exp(-1.74903 * t) * t**3
               + 1.0707e-7 * np.exp(-0.317636 * t) * t**3
               - 0.161001 * np.exp(-1.74903 * t) * t**2
               + 4.02603e-5 * np.exp(-0.317636 * t) * t**2
               - 0.219912 * np.exp(-1.74903 * t) * t
               + 0.00498889 * np.exp(-0.317636 * t) * t
               - 0.150149 * np.exp(-1.74903 * t)
               + 0.150149 * np.exp(-0.317636 * t))
    elif J == 2 and tau == 3 and alpha == 0.2:
        E_t = (0.360714 * np.exp(-0.782777 * t) * t
               + 0.00436552 * np.exp(-0.28389 * t) * t
               - 0.159084 * np.exp(-0.782777 * t)
               + 0.159084 * np.exp(-0.28389 * t))
    return E_t
base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Unified_time_delay'

t = np.linspace(0, 45, 300)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
results = [
    {"J": 2, "tau": 1, "alpha": 0.2},
    {"J": 5, "tau": 1, "alpha": 0.2},
    {"J": 5, "tau": 3, "alpha": 0.2},
    {"J": 2, "tau": 3, "alpha": 0.2}
]

for result in results:
    J_val = result['J']
    tau_val = result['tau']
    alpha_val = result['alpha']
    E_t = laminarflow(t, J_val, tau_val, alpha_val)
    E_t_normalized = E_t / np.sum(E_t)
    c_out_full = np.convolve(c_0, E_t_normalized, mode="full")
    t_conv_full = np.linspace(t[0] + t[0], t[-1] + t[-1], len(c_out_full))
    valid_indices = t_conv_full <= 45
    t_conv = t_conv_full[valid_indices]
    c_out = c_out_full[valid_indices]
    filename_time = f'time_J{J_val}_tau{tau_val}_alpha{alpha_val}.npy'
    filename_conc = f'concentration_J{J_val}_tau{tau_val}_alpha{alpha_val}.npy'
    np.save(os.path.join(base_dir, filename_time), t_conv)
    np.save(os.path.join(base_dir, filename_conc), c_out)
    print(f"integral E_t: {np.trapz(E_t, t)}")
    ax1.plot(t, E_t, label=f'J={J_val}, tau={tau_val}, alpha={alpha_val}')
    ax2.plot(t_conv, c_out, label=f'J={J_val}, tau={tau_val}, alpha={alpha_val}')
    
ax1.set_xlabel('t')
ax1.set_ylabel('E(t)')
ax1.legend()
ax2.set_xlabel('t')
ax2.set_ylabel('c(t)')
ax2.legend()
plt.tight_layout()
plt.savefig("Unified_time_delay")
plt.show()
