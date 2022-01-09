import numpy as np

x = []
y = []
tmp = []

with open('data.txt') as data:
    lines=data.readlines()

for i in lines:
    yt = i.replace('\n', '')
    tmp.append(yt)

for i in tmp:
    for j in range(len(i)):
        if i[j] == ',':
            s1 = float(i[:j])
            s = float(i[j+1:])
            x.append([s1])
            y.append([s])

x = np.matrix(x)
y = np.matrix(y)

m = len(x)
one_col = np.ones((m, 1))
x_2 = np.square(x)
X = np.hstack((one_col, x, x_2))
T = np.random.rand(3, 1)


alpha = float(input('Enter alpha value '))
n = int(input('enter number of iterations '))

########COST FUNCTION _________________________________


def Cost(X, y, theta):
    J = np.sum((np.square((np.matmul(X,theta))-y)))/(2*m)
    return J


def Gradient_descent(x, theta, alpha):
    h0 = np.matmul(x,theta)
    for i in range(len(theta)):
        if i < 2:
            d = np.sum(np.multiply((h0 - y), x[:, i])) * (alpha / m)
        else:
            d = np.sum(np.multiply((h0 - y), x[:, i])) * (2 * alpha / m)

        theta[i] = float(theta[i]) - d

    return theta

def iterations(n):

    for i in range (n):
        Gradient_descent(X,T,alpha)
        print('Cost', Cost(X, y, T))




iterations(n)
print('Thetanew',T)
print('Cost', Cost(X, y, T))


