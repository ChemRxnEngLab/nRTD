# %%
import numpy as np
import numpy.typing as npt
from scipy.signal import unit_impulse
import matplotlib.pyplot as plt
from typing import Literal

#plt.style.use("ICIWstyle")


# %%
def my_dirac(t: npt.NDArray[np.float_], idx: int | Literal["mid"] = 0):
    """generates a dirac pulse with area 1 at the given index.

    Parameters
    ----------
    t : np.array
        time vector
    idx : float | Literal["mid"], optional
        idx at which position pulse is generated, by default 0.
        If "mid", peak will be at n//2 (in th emiddle of the array)

    Returns
    -------
    np.array
        array containing the dirac pulse
    """
    if idx == "mid":
        idx = t.size // 2
    d = unit_impulse(t.size, idx)
    d /= np.trapz(d, t)
    return d


# %%
def my_tank_system(t: npt.NDArray, a: float, b: float, tau: float) -> npt.NDArray:
    k = (1 - a) / (b * tau)
    E = a * my_dirac(t)
    E += (1 - a) * k * np.exp(-k * t)
    return E


def tank_system(t: npt.NDArray, a: float, b: float, tau: float) -> npt.NDArray:
    k = (1 - a) / (b * tau)
    E = a * unit_impulse(t.size)
    E += (1 - a) * k * np.exp(-k * t)
    return E


# %%
t = np.linspace(0, 20, 100, endpoint=True)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1
fig_1, axs = plt.subplots(2)
fig_1.suptitle("My dirac impulse")
ax_1, ax_2 = axs
for i in np.linspace(0, 1, 11, endpoint=True):
    print(f"alpha={i}")
    E = my_tank_system(t, i, 0.5, 5)
    print(E.sum())
    print(np.trapz(E, t))
    ax_1.plot(t, E)
    ax_2.plot(
        np.linspace(0, 100, 2 * t.size - 1),
        np.convolve(c_0, E / np.sum(E), mode="full"),
    )

ax_1.set(
    ylim=(0, 10),
)

fig_2, axs = plt.subplots(2)
fig_2.suptitle("Scipys unit impulse")
ax_1, ax_2 = axs
for i in np.linspace(0, 1, 11, endpoint=True):
    print(f"alpha={i}")
    E = tank_system(t, i, 0.5, 5)
    print(E.sum())
    print(np.trapz(E, t))
    ax_1.plot(t, E)
    ax_2.plot(
        np.linspace(0, 100, 2 * t.size - 1),
        np.convolve(c_0, E / np.sum(E), mode="full"),
    )

ax_1.set(
    ylim=(0, 1.1),
)
plt.show()

plt.show()
