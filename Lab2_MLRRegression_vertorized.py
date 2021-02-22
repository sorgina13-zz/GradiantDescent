
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Lab 2 Linear regression with GD:  many feature 1 target - VECTORIZED

# This function will take in all the feature data X
# as well as the current coefficient and bias values
# It should multiply all the feature value by their associated 
# coefficient and add the bias. It should then return the predicted 
# y values
def hypothesis(X, coefficients, bias):
    
    predictedY = np.zeros(X.shape[0])
    
    # TODO: Calculate and return predicted results
    
    predictedY = np.dot(coefficients , X.T) + bias
    
    return predictedY




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
        
        # TODO: 
        # Calculate predicted y values for current coefficient and bias values 
        # calculate and update bias using gradient descent rule
        # Update each coefficient value in turn using gradient descent rule

        
        predictedY = hypothesis(X, coefficients, bias)
        
        biasGrad = 1/length * np.sum(predictedY-Y)
        bias = bias - (alpha * biasGrad)
        
        coeffGrad = (1/length) * ((predictedY -Y)@X)
        coefficients = coefficients - (alpha * coeffGrad)
                    
        
        loss = (1.0/(2*length))*(np.sum( (predictedY - Y)**2))
        
        errorValues.append(loss)
        
    # calculate R squared value for current coefficient and bias values
    rSquared = calculateRSquared(bias, coefficients,X, Y)
    print ("Final R2 value is ", rSquared)
    print("Learning rate : ",alpha)
    print("Number of Iterations : ", num)


    # plot the cost for each iteration of gradient descent
    plt.plot(errorValues)
    plt.show()
    
    return bias, coefficients



# Complete this function for part 2 of the exercise. 

def calculateTestR2(bias, coefficients, testFile):
    
    testFile = testFile.dropna()
    
    data = testFile.values
    data.shape
        # Seperate teh features from the target feature    
    Y = data[:, -1]
    X = data[:, :-1]
    
    # Standardize each of the features in the dataset. 
    for num in range(len(X[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature))/np.std(feature)
        X[:, num] = feature
        

    
    predictedY = hypothesis(X, coefficients, bias)
    
    avgY = np.average(Y)
    totalSumSq = np.sum((avgY - Y)**2)
    
    sumSqRes = np.sum((predictedY - Y)**2)
    
    r2 = 1.0-(sumSqRes/totalSumSq)
    
    return r2
    

def multipleLinearRegression(X, Y):

    # set the number of coefficients (weights) equal to the number of features
    #complete this line of code
    coefficients = np.zeros(X.shape[1])
    
    bias = 0.0
    
    alpha = 0.1 # learning rate
    
    max_iter= 100
    


    # call gredient decent, and get intercept(=bias) and coefficents
    bias, coefficients = gradient_descent(bias, coefficients, alpha, X, Y, max_iter)
    
    return bias, coefficients
    
    
    
def main():
    tic = time.process_time() 
    
    df = pd.read_csv("./Dataset/trainingData.csv")
    df = df.dropna()
    #print (df.shape)

    
    data = df.values

     
    # Seperate teh features from the target feature    
    Y = data[:, -1]
    X = data[:, :-1]
    
    # Standardize each of the features in the dataset. 
    for num in range(len(X[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature))/np.std(feature)
        X[:, num] = feature
        
    print(X.shape)
    print(Y.shape)
     
    # run regression function and return bias and coefficients (weights)
    bias, coefficients = multipleLinearRegression(X, Y)
    
    # Enable code if you have a test set  (as in part 2)
    testFile = pd.read_csv("./Dataset/testData.csv")

    
    calculateTestR2(bias, coefficients, testFile)
    toc = time.process_time() 
    n_tic = time.process_time() 
    print("process time: ", n_tic)
    

    

main()
