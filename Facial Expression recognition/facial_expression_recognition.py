
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
from functions import getbinary
from sklearn.utils import shuffle

class LogisticModel(object):
    def __init__(self):
        pass
    def fit(self,X,Y,X_test,Y_test,lr = 10e-7,reg = 10e-22,epoch=120000,show_fig=False):
        N,D = X.shape
        self.W=(np.random.randn(D)/np.sqrt(D)).reshape(D,1)
        #Running gradient descent now
        cost=[]
        for i in range(0,epoch):
            y_pred = self.forward(X)
            self.W = self.W-lr*(X.T.dot(y_pred-Y)+reg*self.W)
            #b -= learning_rate*((pY - Y).sum() + reg*b)

            if i%20 ==0:
                yhat = self.forward(X_test)
                c = self.cross_entropy(Y_test,yhat)

                cost.append(c)
                r = self.classification_rate(Y_test,np.round(yhat))

                print("i:", i , "cost:" ,c,"rate:",r)

    def sigmoid(self,Z):
        Z =np.exp(-Z)
        return 1/(1+Z)
    def forward(self,X):
        return self.sigmoid(X.dot(self.W))
    def classification_rate(self,T,Y):
        return np.mean(T==Y)
    def cross_entropy(self,T,Y):
        return-(T*np.log(Y)+(1-T)*np.log(1-Y)).sum()



def main():
    X,Y = getbinary() #we extract binary data from the complete dataset using the function getbinary
    Y=Y.reshape(Y.shape[0],1)
    count1 = np.count_nonzero(Y) # This means we have only 547 samples of class 1 and 4953 samples of class 0
    # This creates a class imbalance problem and we have to address that by repeating the copies of the class 1 data set atleast 9 folds of the current dataset
    X_class1 =[]
    for i in range(0,Y.shape[0]):
        temp = Y[i]
        if temp==1:
            X_class1.append(X[i,:])


    X_class1=np.repeat(X_class1,8,axis=0) # Repeats ndarray 8 times and stacks it vertically along rows
    #So we now are going to add 4376 additional elements in X matrix and subsequently we need to add similar elements in Y

    X = np.vstack((X,X_class1))
    Y_class1 = np.ones((X_class1.shape[0],1))
    Y = np.vstack([Y,Y_class1])
    X,Y = shuffle(X,Y)
    # Now we have 4923 samples of class 1 and 4953 samples of class 0 , so we have sorted the class imbalance problem
    X_test,Y_test = X[-1000:,:],Y[-1000:,:] # Converting into train and test sets
    X,Y=X[:-1000,:],Y[:-1000,:]
    bias = np.ones((X.shape[0],1))
    X = np.hstack([bias,X])
    bias2 =np.ones((X_test.shape[0],1))
    X_test =np.hstack([bias2,X_test])

    model = LogisticModel()
    model.fit(X,Y,X_test,Y_test,show_fig=True)

if __name__ == '__main__':
    main()
