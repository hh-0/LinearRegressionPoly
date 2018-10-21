import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def flower_to_float(s):
    d = {b'Iris-setosa': 0., b'Iris-versicolor': 1., b'Iris-virginica': 2.}
    return d[s]

irises = np.loadtxt('iris.data', delimiter=',', converters = {4: flower_to_float})

data = irises[:,:4]
raw_labels = irises[:,4]

labels = np.zeros((raw_labels.shape[0], 3))
labels[np.arange(raw_labels.shape[0]), raw_labels.astype(int)] = 1

data = np.column_stack((np.ones((data.shape[0],1)),data))

temp_data = np.column_stack((data, labels))
np.random.shuffle(temp_data)

d = int(temp_data.shape[0]/2)
train_data = temp_data[:d, :5]
train_label = temp_data[:d, 5:]
test_data = temp_data[d:, :5]
test_label = temp_data[d:, 5:]

def f(w):
    priors = (0.0005/2) * w.dot(w)
    likelihoods = 0
    for n in range(train_data.shape[0]):
        a = None
        b = None
        for k in range(3):
            if a is None:
                a = train_label[n,k] * w[k*5:(k+1)*5].dot(train_data[n])
                b = np.exp(w[k*5:(k+1)*5].dot(train_data[n]))
            else:
                a += train_label[n,k] * w[k*5:(k+1)*5].dot(train_data[n])
                b += np.exp(w[k*5:(k+1)*5].dot(train_data[n]))
        likelihoods += a - np.log(b)
    return priors - likelihoods

w_init = np.ones(15)
w_hat = scipy.optimize.minimize(f, w_init).x

def predict(w, test_set):
    result = []
    d = 0
    
    for n in range(test_set.shape[0]):
        b = None
        for k in range(3):
            if b is None:
                b = np.exp(w[k*5:(k+1)*5].dot(test_set[n]))
            else:
                b += np.exp(w[k*5:(k+1)*5].dot(test_set[n]))
        best = [-999, -999]
        for k in range(3):
            a = np.exp(w[k*5:(k+1)*5].dot(test_set[n]))
            if a/b > best[1]:
                best[1] = a/b
                best[0] = k
        result.append(best)
    return np.array(result)[:,0]

pred = predict(w_hat, test_data)
test_label = np.array([np.where(r==1)[0][0] for r in test_label])

correct = 0
total = test_label.shape[0]
for i in range(total):
    if pred[i] == test_label[i]:
        correct += 1

print(correct)
print(correct/total)