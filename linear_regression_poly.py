import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

data = np.loadtxt('crash.txt')

train_set = data[0::2]
test_set = data[1::2]

def basis_functions(X, L):
    w = np.empty(L+1)
    for x in X:
        p = [x ** (order) for order in range(L+1)]
        w = np.vstack((w, p))
    return w[1:]

x_train = train_set[:,0]
t_train = train_set[:,1]
t_train = t_train.reshape((-1,1))
test_x = test_set[:,0]
test_t = test_set[:,1].reshape((-1,1))

x_plot = list(range(1,21))
y_train = []
y_test = []
lowest = 1000
L_lowest = 1
for L in range(1,21):
    phi_train = basis_functions(x_train, L)
    w = np.linalg.solve(phi_train.T.dot(phi_train), phi_train.T.dot(t_train))

    E_train = t_train - phi_train.dot(w)
    E_train = E_train ** 2
    RMS_train = np.sqrt(E_train.sum()/x_train.shape[0])

    phi_test = basis_functions(test_x, L)
    E_test = test_t - phi_test.dot(w)
    E_test = E_test ** 2
    RMS_test = np.sqrt(E_test.sum()/test_x.shape[0])

    if RMS_test < lowest:
        lowest = RMS_test
        L_lowest = L

    y_train.append(RMS_train)
    y_test.append(RMS_test)

plt.subplot(2, 1, 1)
plt.title('RMS')
plt.xticks(x_plot)
plt.plot(x_plot, y_train, label='Training')
plt.plot(x_plot, y_test, label='Testing')
plt.legend()

phi_train = basis_functions(x_train, 14)
w = np.linalg.solve(phi_train.T.dot(phi_train), phi_train.T.dot(t_train))
pred_train = phi_train.dot(w)
phi_test = basis_functions(test_x, 14)
pred_test = phi_test.dot(w)

plt.subplot(2, 1, 2)
plt.title('Best Fit')
plt.scatter(x_train, t_train, c='black', label='Training Data')
plt.scatter(test_x,test_t, c='green', label ='Testing Data')
plt.plot(x_train, pred_train, label='Training Best Fit')
plt.plot(test_x, pred_test, c='red', label='Testing Best Fit')
plt.legend()
plt.show()