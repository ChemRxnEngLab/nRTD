import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import os

def dispersion(t: npt.NDArray[np.float64], Bo: float, tau: float) -> npt.NDArray[np.float64]:
    print(f"tau = {tau}")
    teta=t/tau
    # E[t==0]=0 #I did its zero before calc to avoid the error
    E = np.zeros_like(t)
    non_zero_indices = t > 0
    teta_non_zero = teta[non_zero_indices]
    E[non_zero_indices] = (1/2) * (np.sqrt(Bo / (np.pi * teta_non_zero))) * np.exp(-(Bo * ((1 - teta_non_zero) ** 2)) / (4 * teta_non_zero))
    E = E / tau
    return E

t = np.linspace(0, 300, 200, endpoint=True) #last position
c_0 = np.zeros_like(t)
c_0[t > 5] = 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

Bo_values = np.array([1])

base_dir = r'D:\Tuana\nRTD\Experiments\Preliminary\Litrature\Dispersion_Model'
for Bo in Bo_values:
    E = dispersion(t, Bo, 5) #Def. the parameters, tau=5
    print(f"Bo: {Bo}, E= {np.trapz(E, t)}")
    c_out_full = np.convolve(c_0, E / np.sum(E), mode="full")
    t_conv_full = np.linspace(t[0] + t[0], t[-1] + t[-1], len(c_out_full))
    valid_indices = t_conv_full <= 300
    t_conv = t_conv_full[valid_indices]
    c_out = c_out_full[valid_indices]
    Bo_dir = os.path.join(base_dir, f'Bo{Bo}200dis_300s_1412')
    os.makedirs(Bo_dir, exist_ok=True)
    print("c_out",c_out.shape)
    print("t_conv_full",t_conv.shape)
    
    np.save(os.path.join(Bo_dir, 'time_300.npy'), t_conv)
    np.save(os.path.join(Bo_dir, 'concentration_300.npy'), c_out)

    