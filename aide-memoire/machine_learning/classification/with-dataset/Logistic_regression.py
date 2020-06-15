import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.loadtxt('ex2data1.txt',delimiter = ',')
Exam1 = data[:,0]
Exam2 = data[:,1]
X = np.array([Exam1, Exam2]).T

y = data[:,2]

idx_1 = np.where(y == 1)
idx_0 = np.where(y == 0)

fig1 = plt.figure(figsize=(5,5)) 
ax = plt.axes() 
ax.set_aspect(aspect = 'equal', adjustable = 'box')
plt.title('Distribution')
plt.grid(axis='both', alpha=.25)
ax.scatter(Exam1[idx_1], Exam2[idx_1], s=50, c='b', marker='o', label='Admitted')
ax.scatter(Exam1[idx_0], Exam2[idx_0], s=50, c='r', marker='*', label='Not Admitted')
plt.show()

def sigmoid(z):
    """
    return the sigmoid of z
    """
    
    return 1/ (1 + np.exp(-z))

sigmoid(0)

def hypothesis(X, theta):
    return sigmoid(np.dot(X,theta))
    
def gradient(theta, X, y):
     predictions = sigmoid(np.dot(X,theta))
     return 1/m * np.dot(X.transpose(),(predictions - y))
    
    
def costFunction(theta, X, y):
    """
    Takes in numpy array theta, x and y and return the logistic regression cost function and gradient
    """
    
    m=len(y)
    
    predictions = sigmoid(np.dot(X,theta))
    error = -((y * np.log(predictions)) + ((1-y)*np.log(1-predictions)))

    cost = 1/m * sum(error)
    
    grad = 1/m * np.dot(X.transpose(),(y - predictions))
    
    return cost[0] , grad

def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    
    X_norm = (X - mean)/std
    
    return X_norm , mean , std

m , n = X.shape[0], X.shape[1]
X, X_mean, X_std = featureNormalization(X)
X= np.append(np.ones((m,1)),X,axis=1)
y=y.reshape(m,1)
initial_theta = np.zeros((n+1,1))
cost, grad= costFunction(initial_theta,X,y)
print("Cost of initial theta is",cost)
print("Gradient at initial theta (zeros):",grad)

def log_Likelihood(theta, X,y):
    h = sigmoid(np.dot(X,theta))
    return np.sum((y * np.log(h)) + ((1-y)*np.log(1-h)))
    
    
def gradientDescent(X,y,theta,alpha,num_iters):     
    m=len(y)
    J_history =[]   
    for i in range(num_iters):
        cost, grad = costFunction(theta,X,y)
        theta = theta + (alpha * grad)
        J_history.append(cost)        
    return theta , J_history

theta , J_history = gradientDescent(X,y,initial_theta,.05,3000)

print("Theta optimized by gradient descent:",theta)
print("The cost of the optimized theta:",J_history[-1])

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()

print(theta)


fig1 = plt.figure(figsize=(5,5)) 
ax = plt.axes() 
#ax.set_aspect(aspect = 'equal', adjustable = 'box')
plt.title('Logistic regression')
plt.grid(axis='both', alpha=.25)
ax.scatter(X[:,1][idx_1], X[:,2][idx_1], s=50, c='b', marker='o', label='Admitted')
ax.scatter(X[:,1][idx_0], X[:,2][idx_0], s=50, c='r', marker='*', label='Not Admitted')
x_value= np.array([np.min(X[:,1]),np.max(X[:,1])]) 
y_value=-(theta[0] +theta[1]*x_value)/theta[2] #+ Exam2.mean() + (Exam2.std())
plt.plot(x_value,y_value, "g")
plt.legend(loc=0)
plt.show()


X_new = np.array([Exam1, Exam2]).T
X_new= np.append(np.ones((m,1)),X_new,axis=1)
print(theta)


fig1 = plt.figure(figsize=(5,5)) 
ax = plt.axes() 
#ax.set_aspect(aspect = 'equal', adjustable = 'box')
plt.title('Logistic regression')
plt.grid(axis='both', alpha=.25)
ax.scatter(X_new[:,1][idx_1], X_new[:,2][idx_1], s=50, c='b', marker='o', label='Admitted')
ax.scatter(X_new[:,1][idx_0], X_new[:,2][idx_0], s=50, c='r', marker='*', label='Not Admitted')
x_value= np.array([np.min(X_new[:,1]),np.max(X_new[:,1])])
y_value=-(theta[0] +theta[1]*x_value)/theta[2]  
difference =  np.array([max(X_new[:,2]) - y_value[0], min(X_new[:,2]) - y_value[1]])
y_value = y_value + difference
plt.plot(x_value,y_value, "g")
plt.legend(loc=0)
plt.show()


def classifierPredict(theta,X):    
    predictions = X.dot(theta)    
    return predictions>0

x_test = np.array([45,85])
x_test = (x_test - X_mean)/X_std
x_test = np.append(np.ones(1),x_test)
prob = sigmoid(x_test.dot(theta))
print("For a student with scores 45 and 85, we predict an admission probability of",prob[0])

p=classifierPredict(theta,X)
print("Train Accuracy:", sum(p==y)[0],"%")


x_test = np.array([85,85])
x_test = (x_test - X_mean)/X_std
x_test = np.append(np.ones(1),x_test)
prob = sigmoid(x_test.dot(theta))
print("For a student with scores 85 and 85, we predict an admission probability of",prob[0])

x_test = np.array([50,50])
x_test = (x_test - X_mean)/X_std
x_test = np.append(np.ones(1),x_test)
prob = sigmoid(x_test.dot(theta))
print("For a student with scores 50 and 50, we predict an admission probability of",prob[0])


x_test = np.array([10,90])
x_test = (x_test - X_mean)/X_std
x_test = np.append(np.ones(1),x_test)
prob = sigmoid(x_test.dot(theta))
print("For a student with scores 10 and 90, we predict an admission probability of",prob[0])