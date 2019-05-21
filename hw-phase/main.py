# 信号与数据

import matplotlib.pyplot as plt
import numpy as np

n = 128
x = np.random.randn(n,1) + 1j * np.random.rand(n,1)

m = round(4.5*n)
A = 1 / np.sqrt(2) * np.random.randn(m,n) + 1j / np.sqrt(2) * np.random.randn(m,n)
y = np.abs(A @ x)**2

# 初始化

iteration = 50
z0 = np.random.randn(n,1)
z0 /= np.linalg.norm(z0)

for _ in range(iteration):
    z0 = np.conj(A.T)@(np.diag(y.flatten())@(A@z0))
    z0 /= np.linalg.norm(z0)

normest = np.sqrt(np.sum(y)/y.size)
z = normest * z0
complx = np.trace(np.conj(x.T)@z)
Relerrs = [np.linalg.norm(x - np.exp(-1j*np.arctan(complx.imag/complx.real))*z)/np.linalg.norm(x)]

T = 2500
tau0 = 330
def mu(t):
    return min(1 - np.exp(-t/tau0),0.2)

## 迭代

for t in range(T):
    yz = A@z
    grad = np.conj(A.T)@((np.abs(yz)**2-y)*yz) * (1/m) # Wirtinger 梯度
    z -= ((mu(t))/(normest**2)) * grad
    complx = np.trace(np.conj(x.T)@z)
    Relerrs.append(np.linalg.norm(x-np.exp(-1j * np.arctan(complx.imag / complx.real))*z)/np.linalg.norm(x))
    if not grad.any():
        break
print('Relative error after initialization: ',Relerrs[0])
print('Relative error after {} iterations: {}\n'.format(str(T),str(Relerrs[T])))

plt.title('Experiment 3')
plt.xscale('linear')
plt.yscale('log')
plt.plot(Relerrs)
plt.grid()
plt.show()

# Relative error after initialization:  1.0489348203835522
# Relative error after 2500 iterations: 3.9402815057984826e-15

# Relative error after initialization:  1.03898179977986
# Relative error after 2500 iterations: 1.4300247639334214e-15

# Relative error after initialization:  1.0162443311529112
# Relative error after 2500 iterations: 4.772241069328224e-14
