import numpy as np
n_0_in=100
n_0_E=99
n_0_out=200
n_1_in=n_0_out
n_1_E=132
n_1_out=333

### Model
t_0_in = np.linspace(0,50,... )
c_0_in = np.zeros(...) 
c_0_in[t_0_in>5] = 1

c_1 = np.convolve(lam_flow_model, c_0_in)
## len(c_1)= n_0_out

c_2 = np.convolve(Adler_havarka, c_1)
## len(c_2)= n_1_out

### Neural Network
#input to the first NNlayer
c_0_in = np.zeros((n_0_in,))
c_0_in[t_0_in>5] = 1

model_1= RTDModule(
    kernel_size=n_0_E,
    ...
)

ds_1 = TensorDataset(torch.tensor(c_0_in), torch.tensor(c_1))

Triner.train(model_1, ds_1)

model_1.E
plt.plot()

c_1_tilde = model_1(torch.tensor(c_0_in))

#input to the second NNlayer

model_2 = RTDModule(
    kernel_size=n_1_E,
    ...
)

ds_2 = TensorDataset(c_1_tilde, torch.tensor(c_2))

Trainer.train(model_2, ds_2)

model2.E
plt.plot()