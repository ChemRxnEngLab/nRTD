import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import os

def dispersion(t: npt.NDArray[np.float64], Bo: float, tau: float) -> npt.NDArray[np.float64]:
    print(f"tau = {tau}")
    teta=t/tau
    # E[t==0]=0 #I did its zero before calc to avoid the error
    E = np.zeros_like(t, dtype=np.float64)
    non_zero_indices = t > 0
    teta_non_zero = teta[non_zero_indices]
    E[non_zero_indices] = (1/2) * (np.sqrt(Bo / (np.pi * teta_non_zero))) * np.exp(-(Bo * ((1 - teta_non_zero) ** 2)) / (4 * teta_non_zero))
    E = E / tau
    return E

t = np.linspace(0, 100, 500, endpoint=True)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

Bo_values = np.array([1,3,10])

base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Dispersion_Model'

for Bo in Bo_values:
    E = dispersion(t, Bo, 5) #Def. the parameters, tau=5
    print(f"Bo: {Bo}, E= {np.trapz(E, t)}")
    c_out_full = np.convolve(c_0, E / np.sum(E), mode="full")
    t_conv_full = np.linspace(t[0] + t[0], t[-1] + t[-1], len(c_out_full))
    valid_indices = t_conv_full <= 100
    t_conv = t_conv_full[valid_indices]
    c_out = c_out_full[valid_indices]
    Bo_dir = os.path.join(base_dir, f'Bo{Bo}')
    os.makedirs(Bo_dir, exist_ok=True)
    
    np.save(os.path.join(Bo_dir, 'time.npy'), t_conv)
    np.save(os.path.join(Bo_dir, 'concentration.npy'), c_out)

    ax1.plot(t, E, label=f'Bo {Bo:}')
    ax2.plot(
        np.linspace(0, 100, len(c_out)),
        c_out,
        label=f'Bo {Bo:}'
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
plt.savefig('dispersion_model_001.png', dpi=300)
plt.show()

t_conv = np.load(os.path.join(Bo_dir, 'time.npy'))
c_out = np.load(os.path.join(Bo_dir, 'concentration.npy'))

for Bo in Bo_values:
    Bo_dir = os.path.join(base_dir, f'Bo{Bo}')
    t_conv = np.load(os.path.join(Bo_dir, 'time.npy'))
    c_out = np.load(os.path.join(Bo_dir, 'concentration.npy'))
    print(f"Time array shape: {t_conv.shape}")
    print(f"Concentration array shape: {c_out.shape}")

print(f"\nResults for Bo = {Bo:}:")
print(f"Time: {t_conv[:10]}...")
print(f"Concentration: {c_out[:10]}...")
