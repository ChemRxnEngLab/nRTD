import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

def zeta_func(beta: float, R: float) -> float:
    return np.sqrt((1 + R) / (2 * beta * (1 - beta)))

def tank_recycle(t: npt.NDArray[np.float64], beta: float, R: float, tau: float) -> npt.NDArray[np.float64]:
    zeta = zeta_func(beta, R)
    exponential_term = np.exp(-t / (4 * zeta**2 * tau))
    sqrt_term = np.sqrt(zeta**2 - 1)
    E = (zeta / sqrt_term) * exponential_term / tau * (
        np.exp(2 * t * zeta * sqrt_term / tau) - np.exp(-2 * t * zeta * sqrt_term / tau))
    return E

t = np.linspace(0, 20, 1000, endpoint=True)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1
fig, axs = plt.subplots(2)
ax1, ax2 = axs

beta_val = np.linspace(0.1, 0.5, 5)
R_val = 0.1
tau_val = 5
for beta in beta_val:
    E = tank_recycle(t, beta, R_val, tau_val) 
    ax1.plot(t, E, label=f'Beta={beta:.2f}, R={R_val:.2f}, Tau={tau_val}')
    ax2.plot(
        np.linspace(0, 40, 2 * t.size - 1),
        np.convolve(c_0, E / np.sum(E), mode="full"),
        label=f'Beta={beta:.2f}, R={R_val:.2f}, Tau={tau_val}'
    )

ax1.set_xlabel("t")
ax1.set_ylabel("E")
ax1.legend()
ax2.set_xlabel("t")
ax2.set_ylabel("C")
ax2.legend()
plt.savefig('tankrecycle_model.png', dpi=300)
plt.show()
