import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import os

def tank_recycle(t: npt.NDArray[np.float64], beta: float, R: float, tau: float) -> npt.NDArray[np.float64]:
    print(f"tau = {tau}")
    print(f"R = {R}")
    zeta = (np.sqrt (1 + R)) / (2* (np.sqrt (beta * (1 - beta))))
    exponential_term = np.exp(-t / (4 * (zeta**2) * tau))
    sqrt_term = np.sqrt((zeta**2) - 1)
    E = (zeta / sqrt_term) * (exponential_term / tau) * (
        np.exp((2 * t * zeta * sqrt_term )/ tau) - np.exp((-2 * t * zeta * sqrt_term )/ tau))
    return E

t = np.linspace(0, 20, 1000, endpoint=True)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

beta_values = np.array([0.3,0.55,0.7])

base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Tank_recycle_Model'

for beta in beta_values:
    E = tank_recycle(t, beta, 0.1,30) #Def. the parameters, tau=5
    print(f"beta: {beta}, E= {np.trapz(E, t)}")

    c_out_full = np.convolve(c_0, E / np.sum(E), mode="full")
    t_conv_full = np.linspace(t[0] + t[0], t[-1] + t[-1], len(c_out_full))
    valid_indices = t_conv_full <= 100
    t_conv = t_conv_full[valid_indices]
    c_out = c_out_full[valid_indices]
    Bo_dir = os.path.join(base_dir, f'beta{beta}')
    os.makedirs(Bo_dir, exist_ok=True)
    
    np.save(os.path.join(Bo_dir, 'time.npy'), t_conv)
    np.save(os.path.join(Bo_dir, 'concentration.npy'), c_out)

    ax1.plot(t, E, label=f'beta {beta:}')
    ax2.plot(
        np.linspace(0, 100, len(c_out)),
        c_out,
        label=f'beta {beta:}'
    )
ax1.set_xlabel('t')
ax1.set_ylabel('E')
ax1.legend()

ax2.plot(t, c_0, label='c_0', linestyle='--', color='black')
ax2.set_xlim(0,120)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('t')
ax2.set_ylabel('C')
ax2.legend()

plt.tight_layout()
plt.savefig('tank_recyce_model_001.png', dpi=300)
plt.show()

t_conv = np.load(os.path.join(Bo_dir, 'time.npy'))
c_out = np.load(os.path.join(Bo_dir, 'concentration.npy'))

for beta in beta_values:
    Bo_dir = os.path.join(base_dir, f'beta{beta}')
    t_conv = np.load(os.path.join(Bo_dir, 'time.npy'))
    c_out = np.load(os.path.join(Bo_dir, 'concentration.npy'))
    print(f"Time array shape: {t_conv.shape}")
    print(f"Concentration array shape: {c_out.shape}")

print(f"\nResults for beta = {beta:}:")
print(f"Time: {t_conv[:10]}...")
print(f"Concentration: {c_out[:10]}...")




