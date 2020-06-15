import numpy as np
import matplotlib.pyplot as plt


X = 2 * np.random.rand(100,1)
y = 4 +3 * X+np.random.randn(100,1)

fig1 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
plt.plot(X,y,'b.')
#plt.axes().set_aspect('equal', 'box')


def analytical_solution(X_input, y):
    val1 = np.linalg.inv(np.dot(np.transpose(X_input),X_input))
    val2 = np.dot(np.transpose(X_input),y)
    theta = np.dot(val1,val2)
    return theta

theta_optimal = analytical_solution(X, y)

X = np.insert(X, 0, 1, axis=1)
theta_optimal = analytical_solution(X,y)
y_analytical = np.array([theta_optimal[0]*X[:,0] + theta_optimal[1]*X[:,1]]).T
plt.plot(X[:,1],y_analytical,'c-')


def h(X,theta):
    return X.dot(theta)

m = y.shape[0]
def Cost(theta,X,y):
    return (1./(2*m))*(h(X,theta)-y).T.dot(h(X,theta)-y)

def gradient(X,y,theta):
    grad = X.T.dot(h(X,theta)-y)
    return grad

def gradientDescent(X, y, theta, alpha, num_iters):
    J_iter = np.zeros(num_iters)
    theta_iter = np.array([[],[]])
    for iter in np.arange(num_iters):
        theta = theta - alpha*(1/m)*gradient(X,y,theta)
        J_iter[iter] = Cost(theta,X,y)[0][0]
        if iter == 0:
            theta_iter = np.append(theta_iter, theta, axis = 1)
        else:
            theta_iter = np.append(theta_iter, theta, axis = 0)
    return (theta,J_iter, theta_iter)


alpha =0.001
n_iter = 1000

theta = np.random.randn(2,1)

theta_gd,j_iter, theta_iter = gradientDescent(X,y,theta,alpha,n_iter)
y_gd = np.array([theta_gd[0]*X[:,0] + theta_gd[1]*X[:,1]]).T
plt.plot(X[:,1],y_gd,'g-')
plt.show()

fig2 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
plt.plot(np.arange(n_iter), j_iter,'-')
plt.show()


score_predicted = 1-((y-y_gd)**2).sum()/((y-y.mean())**2).sum()
print(score_predicted)

def stocashtic_gradient_descent(X,y,theta,alpha,num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)       
    for it in range(num_iters):
        cost =0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = X[rand_ind,:].reshape(1,X.shape[1])
            y_i = y[rand_ind].reshape(1,1)
            prediction = np.dot(X_i,theta)
            theta = theta -(1/m)*alpha*( X_i.T.dot((prediction - y_i)))
            cost += Cost(theta,X_i,y_i)
        cost_history[it]  = cost
        
    return theta, cost_history

alpha =0.001
n_iter = 1000
theta = np.random.randn(2,1)
theta_gds,j_iter = stocashtic_gradient_descent(X,y,theta,alpha,n_iter)
y_gds = np.array([theta_gds[0]*X[:,0] + theta_gds[1]*X[:,1]]).T
fig8 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
plt.plot(X[:,1],y,'b.')
plt.plot(X[:,1],y_analytical,'c-')
plt.plot(X[:,1],y_gds,'g-')
plt.show()

fig9 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
plt.plot(np.arange(n_iter), j_iter,'-')
plt.show()

score_predicted = 1-((y-y_gds)**2).sum()/((y-y.mean())**2).sum()
print(score_predicted)


def minibatch_gradient_descent(X,y,theta,alpha,n_iter,batch_size):    
    m = len(y)
    cost_history = np.zeros(n_iter)    #n_batches = int(m/batch_size)
    
    for it in range(n_iter):
        cost =0.0
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in range(0,m,batch_size):
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]            
            X_i = np.c_[np.ones(len(X_i)),X_i]           
            prediction = np.dot(X_i,theta)
            theta = theta -(1/m)*alpha*( X_i.T.dot((prediction - y_i)))
            cost += Cost(theta,X_i,y_i)
        cost_history[it]  = cost
        
    return theta, cost_history

alpha =0.001
n_iter = 1000
theta = np.random.randn(3,1)
theta_gdm,j_iter = minibatch_gradient_descent(X,y,theta,alpha,n_iter,20)
y_gdm = np.array([theta_gdm[0]*X[:,0] + theta_gds[1]*X[:,1]]).T

fig10 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
plt.plot(X[:,1],y,'b.')
plt.plot(X[:,1],y_analytical,'c-')
plt.plot(X[:,1],y_gdm,'g-')
plt.show()

fig11 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
plt.plot(np.arange(n_iter), j_iter,'-')
plt.show()

score_predicted = 1-((y-y_gdm)**2).sum()/((y-y.mean())**2).sum()
print(score_predicted)


X_age = []
for i in np.arange(5, 12, 1):    
    #X_age.append(list(i*365 + np.random.randint(1,360, 30)))
    X_age.append(list(i*365 + np.abs(np.random.normal(0.0,1.0,100)*100)))
    #X_age.append(list(i*365 + np.sort(np.abs(np.random.normal(0.0,1.0,100)*100))))
    
X_age = np.transpose(X_age)

#np.savetxt('age.csv', X_age, delimiter = ',', fmt='%d')
data_txt = np.loadtxt('Housing_data.txt',delimiter = ',')
data_csv = np.loadtxt('age.csv',delimiter = ',')

fig3 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
for i in np.arange(5, 5+data_csv.shape[1]):    
    plt.plot(np.repeat(i-5, len(data_csv[:,i-5])),data_csv[:,i-5],'.')
    plt.xlabel('Grade')
    plt.xlim(1, 6)
    plt.ylabel('Age in months')


data_y = np.mean(data_csv,axis = 0)
data_x = np.arange(data_csv.shape[1])
plt.plot(data_x, data_y,'b-')
plt.show()

fig4, ax = plt.subplots(1,3)
fig4.set_figheight(5)
fig4.set_figwidth(15)

fig4.subplots_adjust(left=.2, bottom=None, right=None, top=None, wspace=.2, hspace=.2)
plt1 = plt.subplot(1,3,1)
plt2 = plt.subplot(1,3,2)
plt3 = plt.subplot(1,3,3)

plt1.hist(data_csv[:,0], label='Grade 1', edgecolor='black')
plt1.set_title('Grade 1')
plt1.set_xlabel('Months')
plt1.set_ylabel('Frequency')
plt1.grid(axis='both', alpha=.25)

plt2.hist(data_csv[:,3], label='Grade 4', edgecolor='black')
plt2.set_title('Grade 4')
plt2.set_xlabel('Months')
plt2.set_ylabel('Frequency')
plt2.grid(axis='both', alpha=.25)

plt3.hist(data_csv[:,6], label='Grade 7', edgecolor='black')
plt3.set_title('Grade 7')
plt3.set_xlabel('Months')
plt3.set_ylabel('Frequency')
plt3.grid(axis='both', alpha=.25)

plt.show()


def standard_dev(x_i):    
    # np.sum(x_i - np.mean(x_i))
    square_error = np.sum(np.square(x_i - np.mean(x_i)))
    mean_square_error = square_error/len(x_i)
    return np.sqrt(mean_square_error)


def gaussian_pdf(x_i):    
    # np.sum(x_i - np.mean(x_i))
    dist_from_mean = np.square(x_i - np.mean(x_i))
    #exp_term = np.power(2.71828,-(dist_from_mean)/(2*standard_dev(x_i)))
    exp_term = np.exp(-(dist_from_mean)/(2*standard_dev(x_i)))    
    normalize_term = 1/(standard_dev(x_i)*np.sqrt(np.pi))
    return normalize_term * exp_term

Grade_1_data_sd = standard_dev(data_csv[:,0]) 
print(Grade_1_data_sd)   
print(np.std(data_csv[:,0]))
#print(sc.norm.pdf(data_csv[:,0]))




fig5, ax = plt.subplots(1,3)
fig5.set_figheight(5)
fig5.set_figwidth(15)

fig5.subplots_adjust(left=.2, bottom=None, right=None, top=None, wspace=.2, hspace=.2)
plt1 = plt.subplot(1,3,1)
plt2 = plt.subplot(1,3,2)
plt3 = plt.subplot(1,3,3)

gaussian_dist = gaussian_pdf(data_csv[:,0])
plt1.plot(data_csv[:,0], gaussian_pdf(data_csv[:,0]), 'b.')
plt1.set_title('Grade 1')
plt1.set_xlabel('Months')
plt1.set_ylabel('Frequency')
plt1.grid(axis='both', alpha=.25)

plt2.plot(data_csv[:,3], gaussian_pdf(data_csv[:,3]), 'g.')
plt2.set_title('Grade 4')
plt2.set_xlabel('Months')
plt2.set_ylabel('Frequency')
plt2.grid(axis='both', alpha=.25)

plt3.plot(data_csv[:,6], gaussian_pdf(data_csv[:,6]), 'r.')
plt3.set_title('Grade 7')
plt3.set_xlabel('Months')
plt3.set_ylabel('Frequency')
plt3.grid(axis='both', alpha=.25)

plt.show()
print('Mean ', np.mean(data_csv[:,0]), np.mean(data_csv[:,3]), np.mean(data_csv[:,6]))



fig6 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
plt.plot(data_csv[:,0], gaussian_pdf(data_csv[:,0]), 'b.')
plt.plot(data_csv[:,4], gaussian_pdf(data_csv[:,4]), 'g.')
plt.plot(data_csv[:,6], gaussian_pdf(data_csv[:,6]), 'r.')
plt.show()


#do not use data_csv.mean() and data_csv.std()
mean_repeat = np.repeat(np.array([np.mean(data_csv, axis=0)]), 100, axis=0)
sd_repeat = np.repeat(np.array([np.std(data_csv, axis=0)]), 100, axis=0)
data_csv_zero_mean = (data_csv- mean_repeat)


fig7 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
plt.plot(data_csv_zero_mean[:,0], gaussian_pdf(data_csv_zero_mean[:,0]), 'b.')
plt.plot(data_csv_zero_mean[:,4], gaussian_pdf(data_csv_zero_mean[:,4]), 'g.')
plt.plot(data_csv_zero_mean[:,6], gaussian_pdf(data_csv_zero_mean[:,6]), 'r.')
plt.show()

data_csv_norm = (data_csv- mean_repeat)/sd_repeat
fig8 = plt.figure()
ax = plt.axes()
plt.grid(axis='both', alpha=.25)
plt.plot(data_csv_norm[:,0], gaussian_pdf(data_csv_norm[:,0]), 'b.')
plt.plot(data_csv_norm[:,4], gaussian_pdf(data_csv_norm[:,4]), 'g.')
plt.plot(data_csv_norm[:,6], gaussian_pdf(data_csv_norm[:,6]), 'r.')
plt.show()
