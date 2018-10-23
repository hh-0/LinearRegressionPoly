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
    M = np.linspace(0,np.max(X), L)
    s = M[1] - M[0]
    w = np.empty(L)
    for x in X:
        p = [np.exp((-1*(x - M[i])**2)/(2*(s**2))) for i in range(M.shape[0])]
        w = np.vstack((w, p))
    return w[1:]

best_alpha = 0
best_RMS = 10000
for alpha in np.logspace(-8,0,100):
    phi_train = basis_functions(x_train, 50)
    w = np.linalg.solve(phi_train.T.dot(phi_train) + (alpha/0.0025) * np.identity(50), phi_train.T.dot(t_train))

    phi_test = basis_functions(x_test, 50)
    E_test = (t_test - phi_test.dot(w))**2
    RMS_test = np.sqrt(E_test.sum()/x_test.shape[0])

    if RMS_test < best_RMS:
        best_RMS = RMS_test
        best_alpha = alpha

print(best_alpha)

L_best = 50
phi_train = basis_functions(x_train, L_best)
w = np.linalg.solve(phi_train.T.dot(phi_train)  + (best_alpha/0.0025) * np.identity(L_best), phi_train.T.dot(t_train))
pred_train = phi_train.dot(w)

phi_test = basis_functions(x_test, L_best)
pred_test = phi_test.dot(w)

plt.title('Best Fit with Best Alpha = {0}'.format(best_alpha))
plt.scatter(x_train, t_train, c='red', label='Training Data')
plt.scatter(x_test,t_test, c='green', label='Testing Data')
plt.plot(x_train, pred_train, color='chocolate', label='Training Best Fit')
plt.plot(x_test, pred_test, c='black', label='Testing Best Fit')
plt.legend()
plt.show()
