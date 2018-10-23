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

def basis_functions(X, L):
    M, s = np.linspace(0,np.max(X), L, retstep=True)
    w = np.empty(L)
    for x in X:
        p = [np.exp((-1*(x - M[i])**2)/(2*(s**2))) for i in range(M.shape[0])]
        w = np.vstack((w, p))
    return w[1:]

x_plot = list(range(5,30,5))
y_train = []
y_test = []
for L in range(5,30, 5):
    phi_train = basis_functions(x_train, L)
    w = np.linalg.solve(phi_train.T.dot(phi_train), phi_train.T.dot(t_train))

    E_train = t_train - phi_train.dot(w)
    E_train = E_train ** 2
    RMS_train = np.sqrt(E_train.sum()/x_train.shape[0])

    phi_test = basis_functions(x_test, L)
    E_test = t_test - phi_test.dot(w)
    E_test = E_test ** 2
    RMS_test = np.sqrt(E_test.sum()/x_test.shape[0])

    y_train.append(RMS_train)
    y_test.append(RMS_test)

plt.subplot(2, 1, 1)
plt.title('RMS')
plt.xticks(x_plot)
plt.plot(x_plot, y_train, label='Training')
plt.plot(x_plot, y_test, label='Testing')
plt.legend()

L_best = 10
M, s = np.linspace(0,np.max(x_train), L_best, retstep=True)

phi_train = basis_functions(x_train, L_best)
w = np.linalg.solve(phi_train.T.dot(phi_train), phi_train.T.dot(t_train))
pred_train = phi_train.dot(w)

phi_test = basis_functions(x_test, L_best)
pred_test = phi_test.dot(w)

plt.subplot(2, 1, 2)
plt.title('Best Fit')
plt.scatter(x_train, t_train, c='red', label='Training Data')
plt.scatter(x_test,t_test, c='green', label='Testing Data')
plt.plot(x_train, pred_train, color='chocolate', label='Training Best Fit')
plt.plot(x_test, pred_test, c='black', label='Testing Best Fit')
plt.legend()
plt.show()