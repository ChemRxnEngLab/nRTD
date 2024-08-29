import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_inverse_laplace(coefficients, t_values):
    s, t = sp.symbols('s t')
    results = []  
    for J_val in coefficients['J']: # Loop for each combination
        for tau_val in coefficients['tau']:
            for alpha_val in coefficients['alpha']:
                F_s = ((1 + (1/J_val) * (tau_val * s + alpha_val - alpha_val / (1 + tau_val * s))))**(-J_val) # G function
                f_t = sp.inverse_laplace_transform(F_s, s, t)
                f_t_numeric = sp.lambdify(t, f_t, modules="numpy")
                E_t = f_t_numeric(t_values)
                E_t[0]=2*E_t[0]
                results.append({
                    'J': J_val,
                    'tau': tau_val,
                    'alpha': alpha_val,
                    'E_t': E_t
                })
                integral_E_t = np.trapz(E_t, t_values)
                print(f"integral E: {integral_E_t}") #print
    return results

coefficients = {
    'J': np.array([1,5]),   
    'tau': np.array([1,2]), 
    'alpha': np.array([0.5]) 
}

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
    np.save(os.path.join(unified_dir, 'time.npy'), t_conv)
    np.save(os.path.join(unified_dir, 'concentration.npy'), c_out)

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
plt.savefig('unified_time_delay_001.png', dpi=300)
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

