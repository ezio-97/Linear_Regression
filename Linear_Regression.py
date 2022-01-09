import numpy as np
from matplotlib import pyplot as plt

x = []
y = []
tmp = []
with open('data.txt') as data:
    lines=data.readlines()      #storing each line as an elements of a list

for i in lines:
    yt = i.replace('\n', '')
    tmp.append(yt)              #Replacing the whitespaces

for i in tmp:
    for j in range(len(i)):
        if i[j] == ',':         #Splitting up x and y features
            s1 = float(i[:j])
            s = float(i[j+1:])
            x.append([s1])
            y.append([s])


#Plotting the data

def Plot(x,y):
    plt.scatter(x, y, color = 'g', marker='x')
    plt.title('Scatter Plot')
    plt.ylabel('Profit ')
    plt.xlabel('Population')

    plt.show()

Plot(x,y)


x = np.matrix(x)                #Storing x and y as numpy arrays
y = np.matrix(y)


m = len(x)
one_col = np.ones((m, 1))       #initialising ones vector with dimention m(number of training data)
X = np.hstack((one_col, x))     #initialising the input matrix X with dimention mx2
T = np.random.rand(2, 1)        #initialise a random theta vector of dimention 2


alpha = float(input('Enter alpha value '))                         # learning rate
n = int(input('enter number of iterations '))                      # number of iterations


########COST FUNCTION _________________________________


def Cost(X, y, theta):
    J = np.sum((np.square((np.matmul(X,theta))-y)))/(2*m)       #matrix multiplication of X and T and summing the squared errors over for  all m
    return J

########Gradient descent _______________________________

def Gradient_descent(x, y , theta, alpha ):
    h0 = np.matmul(x,theta)
    for i in range(len(theta)):
        d = np.sum(np.multiply((h0-y),x[:, i]))*(alpha/m)      #derivative terms
        theta[i] = float(theta[i]) - d                         #new theta
    return theta


def iterations(n):
    for i in range(n):
        Gradient_descent(X, y, T, alpha)
        print(Cost(X, y, T))



###########The main function___________________


iterations(n)
print('ThetaNew',T)
print('Cost',Cost(X, y , T))

