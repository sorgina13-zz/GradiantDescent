
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Lab 2 Linear regression with GD:  many feature 1 target - Non-Vectorized

def hypothesis(X, coefficients, bias):
    
    # array of zeros. Length is same as number of training rows.  
    predictedValues = np.zeros(X.shape[0])
    
    # for each feature multiple the X training data by the appropriate 
    # coefficient and add to to predictedvalues
    for num in range(len(coefficients)):
    
        predictedValues += coefficients[num] * X[:, num]
    
    # finally add the current bias to each of the predicted values
    predictedValues += bias
    
    return predictedValues



def calculateRSquared(bias, coefficients,X, Y):
    
    predictedY = hypothesis(X, coefficients, bias)
    
    avgY = np.average(Y)
    totalSumSq = np.sum((avgY - Y)**2)
    
    sumSqRes = np.sum((predictedY - Y)**2)
    
    r2 = 1.0-(sumSqRes/totalSumSq)
    
    return r2
    


        
    

def gradient_descent(bias, coefficients, alpha, X, Y, max_iter):


    length = len(X)
    
    # array is used to store change in cost function for each iteration of GD
    errorValues = []
    
    for num in range(0, max_iter):
        
        # Calculate predicted y values for current coefficient and bias values 
        predictedY = hypothesis(X, coefficients, bias)
        
        # calculate gradient for bias
        biasGrad =    (1.0/(length)) *  (np.sum( predictedY - Y))
        
        #update bias using GD update rule
        bias = bias - (alpha*biasGrad)
        
        # for loop to update each coefficient value in turn
        for coefNum in range(len(coefficients)):
            
            # calculate the gradient of the coefficient
            gradCoef = (1.0/(length))* (np.sum( (predictedY - Y)*X[:, coefNum]))
            
            # update coefficient using GD update rule
            coefficients[coefNum] = coefficients[coefNum] - (alpha*gradCoef)
           
        # calculate asnd store the value of the cost function for current parameters
        cost = (1.0/(2*length))*(np.sum( (predictedY - Y)**2))
        errorValues.append(cost)
    
    # calculate R squared value for current coefficient and bias values
    rSquared = calculateRSquared(bias, coefficients,X, Y)
    print ("Final R2 value is ", rSquared)
    print("Learning rate : ",alpha)
    print("Number of Iterations : ", num)

    # plot the cost for each iteration of gradient descent
    plt.plot(errorValues)
    plt.show()
    
    return bias, coefficients


def calculateTestAccuracy(bias, coefficients, testFile):
    
    df = pd.read_csv(testFile)
    df = df.dropna()
    data = df.values

     
    # Seperate the features from the target feature    
    testY = data[:, -1]
    testX = data[:, :-1]
    
        # Standardize each of the features in the dataset. 
    for num in range(len(testX[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature))/np.std(feature)
        testX[:, num] = feature
    
    rSquared = calculateRSquared(bias, coefficients,testX, testY)
    print ("Test R2 value is ", rSquared)

    


def multipleLinearRegression(X, Y):

    # set the number of coefficients equal to the number of features
    coefficients = np.zeros(X.shape[1])
    bias = 0.0
    
    alpha = 0.1 # learning rate
    
    max_iter=100

    # call gredient decent, and get intercept(=bias) and coefficents
    bias, coefficients = gradient_descent(bias, coefficients, alpha, X, Y, max_iter)
        
    return bias, coefficients

    
    
def main():
    tic = time.process_time() 
    
    df = pd.read_csv("trainingData.csv")
    df = df.dropna()
    data = df.values

     
    # Seperate the features from the target feature    
    Y = data[:, -1]
    X = data[:, :-1]
    
    # Standardize each of the features in the dataset. 
    for num in range(len(X[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature))/np.std(feature)
        X[:, num] = feature
     
    # run regression function
    bias, coefficients = multipleLinearRegression(X, Y)
    
    # Enable code if you have a test set
    testFile = "testData.csv"
    calculateTestAccuracy(bias, coefficients, testFile)
    toc = time.process_time() 
    n_tic = time.process_time() 
    print("process time: ", n_tic)
    
    

main()
