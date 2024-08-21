import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

def dispersion(t: npt.NDArray[np.float64], Bo: float, tau: float) -> npt.NDArray[np.float64]:
    teta=t/tau
    E =1/2*(np.sqrt(Bo/(np.pi*teta)))*np.exp(-(Bo*((1-teta)**2))/(4*teta))
    E[t==0]=0
    E=E/tau
    return E
t = np.linspace(0, 20, 1000, endpoint=True)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
Bo_values = np.array([0.0000001,1,10,100])
for Bo in Bo_values:
    E = dispersion(t, Bo,5)
    print(f"Bo: {Bo}, E= {np.trapz(E, t)}")
    c_out = np.convolve(c_0, E / np.sum(E), mode="full")
    ax1.plot(t, E, label=f'Bo {Bo:.5f}')
    ax2.plot(
        np.linspace(0, 40, len(c_out)),
        c_out,
        label=f'Bo {Bo:.5f}'
    )
# ax1.set_ylim(0, 1.1 * np.max(E))
ax1.set_xlabel('t')
ax1.set_ylabel('E')
ax1.legend()

ax2.plot(t, c_0, label='c_0', linestyle='--', color='black')
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('t')
ax2.set_ylabel('C')
ax2.legend()
plt.tight_layout()
plt.savefig('dispersion_model.png', dpi=300)
plt.show()
