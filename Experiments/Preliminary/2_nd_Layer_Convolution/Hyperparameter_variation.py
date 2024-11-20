import matplotlib.pyplot as plt
import numpy as np
###Epoch Number
x_epoch_1st = [5000, 10000, 15000, 25000]
y_epoch_1st_test = [0.0001012, 2.5e-7, 2.0425e-7, 2.03e-7]
x_epoch_2nd = [5000, 10000, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000,95000]
y_epoch_2nd_test = [0.000015908, 0.000012, 0.0000075899, 0.0000042074, 0.0000034681,0.0000020994, 0.0000016432, 0.0000016752, 0.0000013186, 0.0000010632,7.7e-7]
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].semilogy(x_epoch_1st, y_epoch_1st_test, color='green', marker='s')
axs[0].set_xlabel('Epoch Number')
axs[0].set_ylabel('Test/Loss')
axs[0].text(0.96, 0.62, 'Logarithmic Scale', transform=axs[0].transAxes,
            fontsize=10, color='black', ha='left', va='top', rotation=270)
axs[0].text(0.85, 0.99, '$1^{st} Layer$', transform=axs[0].transAxes,
            fontsize=10, color='black', ha='left', va='top', rotation=0)
axs[0].legend()
axs[1].semilogy(x_epoch_2nd, y_epoch_2nd_test, color='blue', marker='^')
axs[1].set_xlabel('Epoch Number')
axs[1].set_ylabel('Test/Loss')
axs[1].legend()
axs[1].text(0.96, 0.62, 'Logarithmic Scale', transform=axs[1].transAxes,
            fontsize=10, color='black', ha='left', va='top', rotation=270)
axs[1].text(0.85, 0.99, '$2^{snd} Layer$', transform=axs[1].transAxes,
            fontsize=10, color='black', ha='left', va='top', rotation=0)
plt.show()
###LR

x_LR_1st = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
y_LR_1st_test = [2.0466e-7, 2.0728e-7, 2.0365e-7, 2.0371e-7, 2.0516e-7, 6.577e-7, 0.0010252]
y_LR_2nd_test = [2.1943e-6, 2.0644e-6, 1.6785e-6, 1.7443e-6, 1.734e-6, 2.054e-6, 2.6977e-6]
plt.figure(figsize=(10, 6))
plt.loglog(x_LR_1st, y_LR_1st_test, label='1st Layer', color='green', marker='s')
plt.loglog(x_LR_1st, y_LR_2nd_test, label='2nd Layer', color='blue', marker='^')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Test Loss')
plt.legend()
plt.text(0.98, 0.6, 'Logarithmic Scale', transform=plt.gca().transAxes,
         fontsize=10, color='black', ha='left', va='top', rotation=270)
plt.show()
####
x_LR_1st = [  1e-2, 1e-3, 1e-4, 1e-5]
y_LR_1st_test = [  2.0371e-7, 2.0516e-7, 6.577e-7, 0.0010252]
y_LR_2nd_test = [  1.7443e-6, 1.734e-6, 2.054e-6, 2.6977e-6]

fig, axs = plt.subplots(1, 2, figsize=(14, 6))  
axs[0].loglog(x_LR_1st, y_LR_1st_test, color='green', marker='s')
axs[0].set_xlabel('Learning Rate (log scale)')
axs[0].set_ylabel('Test Loss')
axs[0].text(0.96, 0.62, 'Logarithmic Scale', transform=axs[0].transAxes,
            fontsize=10, color='black', ha='left', va='top', rotation=270)
axs[0].text(0.85, 0.99, '$1^{st} Layer$', transform=axs[0].transAxes,
            fontsize=10, color='black', ha='left', va='top', rotation=0)
axs[0].legend()

axs[1].loglog(x_LR_1st, y_LR_2nd_test, color='blue', marker='^')
axs[1].set_xlabel('Learning Rate (log scale)')
axs[1].set_ylabel('Test Loss')
axs[1].text(0.96, 0.62, 'Logarithmic Scale', transform=axs[1].transAxes,
            fontsize=10, color='black', ha='left', va='top', rotation=270)
axs[1].text(0.85, 0.99, '$2^{snd} Layer$', transform=axs[1].transAxes,
            fontsize=10, color='black', ha='left', va='top', rotation=0)
axs[1].legend()
plt.tight_layout()
plt.show()
