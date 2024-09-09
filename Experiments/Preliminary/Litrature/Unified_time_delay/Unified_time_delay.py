import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.typing as npt

def laminarflow(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    #For J=2,tau=1,alpha=0.2
    E_t=3.24642*np.exp(-2.34833*t)*t+0.0392897* np.exp(-0.851669 *t) *t - 0.477252 *np.exp(-2.34833* t) + 0.477252* np.exp(-0.851669 *t)
    #For J=5,tau=1,alpha=0.2
    E_t=(123.224 * np.exp(-5.24709 * t) * t**4 
        + 5.66367e-9 * np.exp(-0.952909 * t) * t**4
        - 6.36342 * np.exp(-5.24709 * t) * t**3
        + 8.68252e-6 * np.exp(-0.952909 * t) * t**3
        - 4.34703 * np.exp(-5.24709 * t) * t**2
        + 0.00108703 * np.exp(-0.952909 * t) * t**2
        - 1.97921 * np.exp(-5.24709 * t) * t
        + 0.0449 * np.exp(-0.952909 * t) * t
        - 0.450448 * np.exp(-5.24709 * t)
        + 0.450448 * np.exp(-0.952909 * t))
    #For J=5,tau=3,alpha=0.2
    E_t=(
        0.507094 * np.exp(-1.74903 * t) * t**4
        + 9.81543e-11 * np.exp(-0.317636 * t) * t**4
        - 0.0785608 * np.exp(-1.74903 * t) * t**3
        + 1.0707e-7 * np.exp(-0.317636 * t) * t**3
        - 0.161001 * np.exp(-1.74903 * t) * t**2
        + 4.02603e-5 * np.exp(-0.317636 * t) * t**2
        - 0.219912 * np.exp(-1.74903 * t) * t
        + 0.00498889 * np.exp(-0.317636 * t) * t
        - 0.150149 * np.exp(-1.74903 * t)
        + 0.150149 * np.exp(-0.317636 * t)
    )
    #For J=2,tau=3,alpha=0.2
    E_t=(
        0.360714 * np.exp(-0.782777 * t) * t 
        + 0.00436552 * np.exp(-0.28389 * t) * t
        - 0.159084 * np.exp(-0.782777 * t)
        + 0.159084 * np.exp(-0.28389 * t)
    )
    return E_t


t_values = np.linspace(0, 5, 500, endpoint=True)
results = compute_inverse_laplace(coefficients, t_values)
c_0 = np.zeros_like(t_values)
c_0[t_values > 5] = 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/unified_time_delay'

for result in results:
    J_val = result['J']
    tau_val = result['tau']
    alpha_val = result['alpha']
    E_t = result['E_t']
    E_t_normalized = E_t / np.sum(E_t)
    c_out_full = np.convolve(c_0, E_t_normalized, mode="full")
    t_conv_full = np.linspace(t_values[0] + t_values[0], t_values[-1] + t_values[-1], len(c_out_full))
    valid_indices = t_conv_full <= 45
    t_conv = t_conv_full[valid_indices]
    c_out = c_out_full[valid_indices]
    unified_dir = os.path.join(base_dir, f'J_{J_val}_tau_{tau_val}_alpha_{alpha_val}')
    os.makedirs(unified_dir, exist_ok=True)
    #np.save(os.path.join(unified_dir, 'time.npy'), t_conv)
    #np.save(os.path.join(unified_dir, 'concentration.npy'), c_out)

    ax1.plot(t_values, E_t, label=f'J={J_val}, tau={tau_val}, alpha={alpha_val}')
    ax2.plot(t_conv, c_out, label=f'J={J_val}, tau={tau_val}, alpha={alpha_val}')

ax1.set_xlabel('t')
ax1.set_ylabel('E(t)')
ax1.legend()

ax2.plot(t_values, c_0, label='c_0', linestyle='--', color='black')
ax2.set_xlim(0,5)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('t')
ax2.set_ylabel('C')
ax2.legend()

plt.tight_layout()
#plt.savefig('unified_time_delay_001.png', dpi=300)
plt.show()

for result in results:
    J_val = result['J']
    tau_val = result['tau']
    alpha_val = result['alpha']
    Bo_dir = os.path.join(base_dir, f'J_{J_val}_tau_{tau_val}_alpha_{alpha_val}')
    t_conv = np.load(os.path.join(Bo_dir, 'time.npy'))
    c_out = np.load(os.path.join(Bo_dir, 'concentration.npy'))
    print(f"Results for J={J_val}, tau={tau_val}, alpha={alpha_val}:")
    print(f"Time: {t_conv[:10]}...")
    print(f"Concentration: {c_out[:10]}...")

