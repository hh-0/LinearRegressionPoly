import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

data = np.loadtxt('crash.txt')

train_set = data[0::2]
test_set = data[1::2]

x_train = train_set[:,0]
t_train = train_set[:,1]
t_train = t_train.reshape((-1,1))

x_test = test_set[:,0]
t_test = test_set[:,1].reshape((-1,1))

M = np.linspace(0,60, 20)
s = M[1] - M[0]
def basis_functions(X, L):
    w = np.empty(L+1)
    for x in X:
        p = np.empty(L+1)
        for i in range(M.shape[0]):
            p[i] = np.exp((-1*(x - M[i])**2)/(2*(s**2)))
        w = np.vstack((w, p))
    return w[1:]

phi_train = basis_functions(x_train, 20)
w = np.linalg.solve(phi_train.T.dot(phi_train), phi_train.T.dot(t_train))
pred_train = phi_train.dot(w)
# phi_test = basis_functions(x_test, 15)
# pred_test = phi_test.dot(w)

plt.scatter(x_train, t_train, c='red')
plt.scatter(x_test,t_test, c='green')
plt.plot(x_train, pred_train)
# plt.plot(x_test, pred_test, c='purple')
plt.show()