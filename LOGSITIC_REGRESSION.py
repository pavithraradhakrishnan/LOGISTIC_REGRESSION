
import numpy as np
import csv
import math

from operator import itemgetter
x_train =[]
y_train = []
test_data =[]
def data_clean(x_train,y_train,test_data):
    with open("akk.csv",encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile,delimiter=',')
        data = [data for data in rows]


    for x in range(len(data)):
        ins = []
        for y in range(3):
            ins.append(float(data[x][y]))
        if(data[x][y+1] == "M"):
            y_train.append(float(1))
        else:
            np.array(y_train.append(float(0)))
        x_train.append(ins)
        print("x_train is" , x_train)

    print(np.array(x_train))
    print(np.array(y_train))
    x = np.array(x_train)
    y = np.array(y_train)
    print("x in data clean,",x)
    print("y in data clean",y)
    #x_train = x
    #y_train = y


    with open("test.csv",encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile,delimiter=',')
        dota = [dota for dota in rows]

    for k in range(len(dota)):
        ins = []
        for l in range(3):
            ins.append(float(dota[k][l]))

        test_data.append(ins)
    test = np.array(test_data)
    #print(test_data)
    return x,y,test

def normalise_data(x):
    " we can normalise by subtracting the maximum value by the minimum value and  then divide by the standard deviation of x or divide by the range"

    max_x = np.max(x,axis = 0)
    min_x = np.min(x,axis = 0)
    normalised_x = 1 - (max_x - x)/(max_x - min_x)
    return normalised_x
'''
def calculate_weighted_sum(X,weights):
    #return sum(x*y for x,y in zip(X,weights))
     print("x is ",X)
     print("Weights is ,",weights)
     return -np.dot(X, weights.T)
'''
def calculate_sigmoid(x,weight):
    h = 1.0 / (1 + np.exp(-np.dot(x, weight.T)))
    return h

def dot_product(a,b):
    return sum(x*y for x,y in zip(a,b))

def calculate_mean(m):
    return (sum(x for x in m))/len(m)
#def transpose(mat):
 #   [[mat[i] for row in mat] for i in range(len(mat[0]))]


def calculate_gradient(weight,x,y):

    #X = X - y.reshape(X.shape[0] -1)
    #print("y in gradient descent",y)
    #print("weight in gradient descent",weight)
    #print("x in gradient descent",x)
    h = calculate_sigmoid(x,weight)
    #print("h in gradient decent",h)
    first_calc = h - y.reshape(x.shape[0], -1)
    final_calc = np.dot(first_calc.T, x)
    #print("grad in gradient",final_calc)
    return final_calc

def calculate_cost(x,weight,y):
    #print("x in cost",x)
    #print("weight in cost",weight)
    h = calculate_sigmoid(x,weight)
    #print("y before squeeze",y)
    y = np.squeeze(y)
    #print("y after squeeze",y)

    h = np.mean((-(y * np.log(h)) - ((1 - y)) * np.log(1 -h)))
    #print("hh is ", h)
    return h


def grad_descent(x, y, weight, learning, gradient):

    cost = calculate_cost(x,weight, y)
    iteration, change = 1, 1
    while (change > gradient):
        #iteration += 1
        prev_cost = cost
        # finding the weights and updating it after every itertation. The loop goes on unti the condition is met
        weight = weight -  (learning * calculate_gradient(weight,x,y))
        cost  =  calculate_cost(x,weight,y)
        #print("cost in grad",cost)

        change = prev_cost - cost
        #print("change in grad", change)
        iteration = iteration + 1
        #print("iteration no is",iteration)

        #h = calculate_sigmoid(x, weight)

        '''
        difference = h - y
        print("y in grad",y)
        gradient = calculate_gradient(difference, h, y)
        weight = minimise_loss(weight, learning, gradient)

        current_cost = calculate_cost(h, y)
        change = prev_cost - current_cost
        '''
    return weight, iteration
def minimise_loss(weight, learning,gradient):
    return weight - learning * gradient

def predict_val(weight,x):
    prob = calculate_sigmoid(x,weight)
    prediction = np.where(prob >= .5, 1, 0)
    return np.squeeze(prediction)
'''
def fit(x,y,no_of_iteration):
    #weights = [1, 1, 1]
    weights = np.matrix(np.zeros(4))
    print("weights")
    print(weights)
    for i in range(no_of_iteration):
        prob = calculate_weighted_sum(x,weights)
        h = calculate_sigmoid(x, prob)


        print("y in fit",y)
        weight,iteration = grad_descent(x,y,weights,0.01,0.001)

        result  = predict_val(x, weight)
        print("Regression coefficients are", weight)
        print("no of iterations are",iteration)
        print("Predicted values are",result)
'''




x,y,test = data_clean(x_train,y_train,test_data)
print("THe given data is",x)
#print("y value after data clean",y)
X =x
#print("Cap X is ",X)

#X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))

X = normalise_data(X)
one_column = np.ones((len(X),1))
print("The normalised input data is",X)
X = np.concatenate((one_column, X), axis = 1)
weight = np.matrix(np.zeros(4))
#print("xin main", X)
#print("y in main", y)
#print("beta in main", weight)
weight , n = grad_descent(X,y,weight,0.01,0.0005)
predict = predict_val(weight,X)
print("the computed theta value is",weight)
#print("difference between y and predicted y")
print("The acutal output for the given input data is where '1' represents male and '0' representd female",y)
print("The predicted output for the given data is",predict)
y_count = np.sum(y == predict)
print("The calculated regression coefficients are", weight)
print("total no of iterations taken", n)
print("The count of the labels predicted",y_count)
#print("y out",y)
#fit(x,y,1)



















