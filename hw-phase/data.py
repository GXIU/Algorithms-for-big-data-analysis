# 信号与数据

import numpy as np 

n = 128
x = np.random.randn(n) + 1j * np.random.rand(n)

m = round(4.5*n)
A = 1 / np.sqrt(2) * np.random.randn(m,n) + 1j / np.sqrt(2) * np.random.randn(m,n)
y = np.abs(A @ x)**2

# 初始化

iteration = 50
z0 = np.random.randn(n,1)
z0 /= np.linalg.norm(z0)

for t in range(iteration):
    z0 = A.T@(y@(A@z0))
    z0 /= np.linalg.norm(z0)

normest = np.sqrt(np.sum(y)/y.size)
z = normest * z0
complx = np.trace(x.T@z
Relerrs = np.linalg.norm(x - np.exp(-1j*np.arctan(complx.imag/complx.real))*z)/np.linalg.norm(x)

T = 2500
tau0 = 330
def mu(t):
    min(1 - np.exp(-t/tau0),0.2)

for i in range(T):
    yz = A@z
    grad = A.T@((np.abs(yz)**2-y)*yz)/m # Wirtinger 梯度
    z = z - mu(t)/normest**2 *grad
    complx = (np.trace(x.T@z)
    Relerrs = [Relerrs, np.linalg.norm(x-np.exp(-1j * np.arctan(complx.imag / complx.real))*z)/np.linalg.norm(x)]

print('Relative error after initialization: ',Relerrs[0])
print('Relative error after %d iterations: %f\n', T,Relerrs(T)) 

