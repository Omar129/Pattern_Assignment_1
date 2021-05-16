import numpy as np
import sys
import math
import random
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def online_training(x,y,lr=1):
    n = x.shape[1]
    epochs_count = 0 
    weights_array = []          #initializing weights array
    delta_array = []            #initializing delta_array which carries weights changes
    #initializing weights with random number from -1 to 1
    weights = np.zeros(n)       
    for i in range(0,n):
        weights[i] = random.random()*random.randint(-1,1)

    delta = np.ones(n)         #initializing delta with ones to enter while loop
    e = sys.float_info.epsilon
    while (norm(delta,1) > e):
        delta = np.zeros(n)    #giving detla zero values at the begining of each loop
        for i in range(0,len(y)):
            if(y[i]*(weights.dot(x[i])) <= 0):          #check condition if they have the same sign or not
                delta = delta - y[i]*x[i]
                delta = delta / len(y)
                
                weights = weights - (lr * delta)     #updating weights by delta
                weights_array.append(weights)        #filling weight array with weights updated
            delta_array.append(norm(delta,1))        

        epochs_count = epochs_count + 1             #counting ephocs "complete loops"
    return weights,delta_array,weights_array,epochs_count
def batch_perceptron(x,y,lr=1):
    n = x.shape[1]
    epochs_count = 0
    weights_array = []                   #initializing weights array
    delta_array = []                     #initializing delta_array which carries weights changes
                        #initializing weights with random number from -1 to 1
    weights = np.zeros(n)
    for i in range(0,n):
        weights[i] = random.random()*random.randint(-1,1)

    delta = np.ones(n)                  #initializing delta with ones to enter while loop
    e = sys.float_info.epsilon
    while (norm(delta,1) > e):
        delta = np.zeros(n)             #giving detla zero values at the begining of each loop
        for i in range(0,len(y)):
            if(y[i]*(weights.dot(x[i])) <= 0):          #check condition if they have the same sign or not
                delta = delta - y[i]*x[i]
        delta = delta / len(y)
        delta_array.append(norm(delta,1))
        weights = weights - (lr * delta)                #updating weights by delta
        weights_array.append(weights)                   #filling weight array with weights updated
        epochs_count = epochs_count + 1                 #counting ephocs "complete loops"
    return weights,delta_array,weights_array,epochs_count
x_2 = np.array([[50,55,70,80,130,150,155,160],[1,1,1,1,1,1,1,1]]).T
y_2 = np.array([1,1,1,1,-1,-1,-1,-1])
x_1 = np.array([[0,255,0,0,255,0,255,255],[0,0,255,0,255,255,0,255],[0,0,0,255,0,255,255,255],[1,1,1,1,1,1,1,1]]).T
y_1 = np.array([1,1,1,-1,1,-1,-1,1])

print("Parameters for online_training method for problem No.1")
w,d_array,w_array,count = online_training(x_2, y_2)
print("Number of epochs = " + str(count))
print("Number of times model weights are updated = " + str(len(w_array)))
plt.scatter(range(0,len(d_array)),d_array)
plt.show()


print("Parameters for batch_perceptron method for problem No.1")
w,d_array,w_array,count = batch_perceptron(x_2, y_2)
print("Number of epochs = " + str(count))
print("Number of times model weights are updated = " + str(len(w_array)))
plt.scatter(range(0,len(d_array)),d_array)
plt.show()


print("Parameters for online_training method for problem No.4")
w,d_array,w_array,count = online_training(x_1, y_1)
print("Number of epochs = " + str(count))
print("Number of times model weights are updated = " + str(len(w_array)))
plt.scatter(range(0,len(d_array)),d_array)
plt.show()



print("Parameters for online_training method for problem No.4")
w,d_array,w_array,count = batch_perceptron(x_1, y_1)
print("Number of epochs = " + str(count))
print("Number of times model weights are updated = " + str(len(w_array)))
plt.scatter(range(0,len(d_array)),d_array)
plt.show()

x,y = make_classification(25,n_features=2,n_redundant = 0,n_informative=1,n_clusters_per_class=1)
mask_for_y = y == 0
y[mask_for_y] = -1
plt.scatter(x[:,0],x[:,1],marker='o',c=y,s=25,edgecolor='k')
plt.show()
print("Parameters for online training method for mask classification")
w,d_array,w_array,count = online_training(x, y)
print("Number of epochs = " + str(count))
print("Number of times model weights are updated = " + str(len(w_array)))
plt.scatter(range(0,len(d_array)),d_array)
plt.show()


training_len = int(0.75*len(y))
x_training = x[0:training_len]
y_training = y[0:training_len]
x_test = x[training_len+1:len(y)]
y_test = y[training_len+1:len(y)]
w,d_array,w_array,count = online_training(x_training, y_training)
x_plot = np.array([np.min(x_training[:, 0] - 3), np.max(x_training[:, 1] + 3)])
y_plot = (-1/w[1]) * (w[0]*x_plot)
plt.scatter(x_training[:,0],x_training[:,1],marker='o',c=y_training,s=25,edgecolor='k')
plt.plot(x_plot,y_plot)
plt.show()

plt.scatter(x_test[:,0],x_test[:,1],marker='o',c=y_test,s=25,edgecolor='k')
plt.plot(x_plot,y_plot)
plt.show()
count = 0
for i in range(0,len(x_test)):

    if(y[i+training_len]*(w.dot(x_test[i])) >=0):
        count=count+1
accuracy = (count/len(x_test)) *100
print("accuracy = " + str(accuracy) + "%")
