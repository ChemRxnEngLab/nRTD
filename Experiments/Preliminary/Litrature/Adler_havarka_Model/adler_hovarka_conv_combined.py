import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_inverse_laplace(coefficients, t_values):
    s, t = sp.symbols('s t')
    alpha = coefficients['alpha_val']
    results = []  
    for tau_a_val in coefficients['tau_a_val']:  # Loop for each combination
        for tau_p_val in coefficients['tau_p_val']:
            for beta_val in coefficients['beta_val']:
                tau_m_val = (beta_val * (1 - alpha)) / alpha
                F_s = (sp.exp(-tau_p_val * s)) / (1 + beta_val + tau_a_val * s - (beta_val / (1 + tau_m_val * s)))
                print(f"E(s)")
                sp.pprint(F_s)
                f_t = sp.inverse_laplace_transform(F_s, s, t)
                f_t_numeric = sp.lambdify(t, f_t, modules="numpy")
                E_t = f_t_numeric(t_values)
                results.append({
                    'tau_a_val': tau_a_val,
                    'tau_p_val': tau_p_val,
                    'tau_m_val': tau_m_val,
                    'beta_val': beta_val,
                    'E_t': E_t
                })
                print(f"Inverse Laplace transform E(t):")
                sp.pprint(f_t)
                integral_E_t = np.trapz(E_t, t_values)
                print(f"Integral of E(t) over the time range: {integral_E_t}")
                
    return results
coefficients = {
    'tau_a_val': np.array([1]),   
    'tau_p_val': np.array([2]), 
    'beta_val': np.array([0.1]), 
    'alpha_val': 0.2           
}

t_values = np.linspace(0, 50, 84, endpoint=True)
results = compute_inverse_laplace(coefficients, t_values)
c_0 = np.zeros_like(t_values)
c_0[t_values > 5] = 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Adler_havarka_Model'
for result in results:
    tau_a_val = result['tau_a_val']
    tau_p_val = result['tau_p_val']
    tau_m_val = result['tau_m_val']
    beta_val = result['beta_val']
    E_t = result['E_t']
    E_t_normalized = E_t / np.sum(E_t)
    c_out_full = np.convolve(c_0, E_t_normalized, mode="full")
    t_conv_full = np.linspace(t_values[0] + t_values[0], t_values[-1] + t_values[-1], len(c_out_full))
    valid_indices = t_conv_full <= 50
    t_conv = t_conv_full[valid_indices]
    c_out = c_out_full[valid_indices]
    unified_dir = os.path.join(base_dir, f'Combined_tau_a_val_{tau_a_val}_tau_p_val_{tau_p_val}_tau_m_val_{tau_m_val}_beta_val_{beta_val}')
    os.makedirs(unified_dir, exist_ok=True)
    np.save(os.path.join(unified_dir, 'time.npy'), t_conv)
    np.save(os.path.join(unified_dir, 'concentration.npy'), c_out)
    ax1.plot(t_values, E_t, label=f'tau_a={tau_a_val}, tau_p={tau_p_val}, tau_m={tau_m_val:.2f}, beta={beta_val}')
    ax2.plot(t_conv, c_out, label=f'tau_a={tau_a_val}, tau_p={tau_p_val}, tau_m={tau_m_val:.2f}, beta={beta_val}')
ax1.set_xlabel('t')
ax1.set_ylabel('E(t)')
ax1.legend()
ax2.set_xticks(np.arange(0, 121, 1))
ax2.plot(t_values, c_0, label='c_0', linestyle='--', color='black')
ax2.set_xlim(0, 120)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('t')
ax2.set_ylabel('C')
ax2.legend()
plt.tight_layout()
plt.savefig('adler_havarka_001_combined.png', dpi=300)
plt.show()
for result in results:
    tau_a_val = result['tau_a_val']
    tau_p_val = result['tau_p_val']
    tau_m_val = result['tau_m_val']
    beta_val = result['beta_val']
    Adler_havarka_dir = os.path.join(base_dir, f'Combined_tau_a_val_{tau_a_val}_tau_p_val_{tau_p_val}_tau_m_val_{tau_m_val}_beta_val_{beta_val}')
    t_conv = np.load(os.path.join(Adler_havarka_dir, 'time.npy'))
    c_out = np.load(os.path.join(Adler_havarka_dir, 'concentration.npy'))
    print(f"Results for tau_a={tau_a_val}, tau_p={tau_p_val}, tau_m={tau_m_val:.2f}, beta={beta_val}:")
    print(f"Time: {t_conv[:100]}...")
    print(f"Concentration: {c_out[:100]}...")
non_zero_indices = np.where(c_out > 0)[0]
if len(non_zero_indices) > 0:
    first_non_zero_index = non_zero_indices[0]
    first_non_zero_time = t_conv[first_non_zero_index]
    first_non_zero_concentration = c_out[first_non_zero_index]
    print(f"first non-zero concentration at time {first_non_zero_time:.4f} = {first_non_zero_concentration:.4f}")


