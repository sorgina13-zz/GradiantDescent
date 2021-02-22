   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Lab 1 LInear regression with GD:  1 feature 1 target


def gradient_descent(bias,lambda1,alpha,X,y, max_iter):
    '''
    X    = Matrix of X
    y    = Vector of Y

    '''
    m = len(y)
    error = []

    for i in range(max_iter):

        
        prediction = (X*lambda1)+bias
        partial = prediction-y
        

        lambda1 = lambda1 -(1/m)*alpha*np.sum(((partial)*X))
        bias = bias-(1/m)*alpha*np.sum(partial)
        
        MSE = (np.sum((partial)**2))/(2*m)
        error.append(MSE)



        
    return bias,lambda1,error

def calculateRSquared(bias, lambda1,X, Y):
    
    predictedY = predictions = X.dot(lambda1)+bias
    
    avgY = np.average(Y)
    totalSumSq = np.sum((avgY - Y)**2)
    
    sumSqRes = np.sum((predictedY - Y)**2)
    
    r2 = 1.0-(sumSqRes/totalSumSq)
    
    return r2


def linearRegression(X, Y):
    
    # set initial parameters for model
    bias = 0
    lambda1 = 0
    
    alpha = 0.1 # learning rate
    max_iter=100

    #TODO
    # call gredient decent to calculate intercept(=bias) and slope(lambda1)
    bias, lambda1, error = gradient_descent(bias, lambda1, alpha, X, Y, max_iter)
    print ('Final bias and  lambda1 values are = ', bias, lambda1, " respecively." )
    r2 = calculateRSquared(bias, lambda1,X, Y)
    print("Number of iterations :" , max_iter)
    print("Learning rate :", alpha)
    print("Final R2 :", r2)
    
    # plot the data and overlay the linear regression model
    yPredictions = (lambda1*X)+bias
    plt.scatter(X, Y)
    plt.plot(X,yPredictions,'k-')
    plt.show()
        # Plot the error values for current apla value
    plt.subplot(2,1,2)
    plt.plot(error)

    
    
    
def main():
    tic = time.process_time()
    
    # Read data into a dataframe
    df = pd.read_excel('data.xlsx',engine="openpyxl")
    df = df.dropna() 


    # Store feature and target data in separate arrays
    Y = df['Y'].values
    X = df['X'].values
    


    # Perform standarization on the feature data
    X = (X - np.mean(X))/np.std(X)
    
    linearRegression(X, Y)
    toc = time.process_time() 
    n_tic = time.process_time() 
    print("process time: ", n_tic)
    


    

main()
