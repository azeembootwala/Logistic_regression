import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Titanic_data_cleaning import get_data
import pandas


def tar2ind(T,D):
    N = len(T)

    ind = np.zeros((N,D))
    for i in range(0,N):
        ind[i,int(T[i])]=1
    return ind
def sigmoid(Z):
    Z = np.exp(-Z)
    return 1/(1+Z)
def soft_max(Z):
    Z = np.exp(Z)
    return Z/Z.sum(axis=1,keepdims=True)
def forward(X,W2,b2,W1,b1):
    Z = sigmoid(X.dot(W1)+b1)
    out = soft_max(Z.dot(W2)+b2)
    return Z,out
def cross_entropy(T,Y):
    return -(T*np.log(Y)).sum()
def classification_rate(T,Y):
    return np.mean(T==Y)
def derivative_W2(Z,T,Y,W,reg):
    return (Z.T.dot(T-Y))+reg*np.sign(W)
def derivative_b2(T,Y):
    return (T-Y).sum(axis=0)
def derivative_W1(Z,T,Y,W2,X,W1,reg):
    dz = ((T-Y).dot(W2.T))*Z*(1-Z)
    return (X.T.dot(dz))+reg*np.sign(W1)
def derivative_b1(Z,T,Y,W2):
    return (((T-Y).dot(W2.T))*Z*(1-Z)).sum(axis=0)


def main():
    X,T = get_data("train")
    X,T = shuffle(X,T)
    N,D = X.shape
    M = 20 #Hidden layer size
    K = 2 # Number of classes
    ind = tar2ind(T,K)
    W1 = np.random.randn(D,M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M,K)
    b2 = np.random.randn(K)
    epoch = 30000
    lr = 10e-4
    reg = 10e-1
    costs=[]
    for i in range(0,epoch):
         hidden,output = forward(X,W2,b2,W1,b1)

         if i%40==0:
            c = cross_entropy(ind,output)
            y_pred = np.argmax(output,axis=1)
            y_pred = y_pred.reshape(y_pred.shape[0],1)
            r = classification_rate(T,y_pred)
            costs.append(c)
            print("cost:",c,"Classi_rate:",r)
         W2 = W2 + lr*derivative_W2(hidden,ind,output,W2,reg)
         b2 = b2 + lr*derivative_b2(ind,output)
         W1 = W1 + lr*derivative_W1(hidden,ind,output,W2,X,W1,reg)
         b1 = b1 + lr*derivative_b1(hidden,ind,output,W2)
    X_test = get_data("")
    g,Yhat = forward(X_test,W2,b2,W1,b1)
    Yhat = np.argmax(Yhat,axis=1) #Yhat =np.round(Yhat)
    Yhat = Yhat.reshape(Yhat.shape[0],1)

    df1 = pandas.read_csv("test.csv")
    Submit = np.asarray(df1.ix[:,"PassengerId"]).reshape(418,1)


    result = np.hstack([Submit,Yhat])
    df_final =pandas.DataFrame({"PassengerID":result[:,0],"Survived":result[:,1]},dtype=int)
    df_final["Survived"]=df_final["Survived"].apply(lambda x: 0 if x==None else x)
    df_final.to_csv("submission.csv",sep=",",index=False,dtype=int)







if __name__ == "__main__":
    main()
