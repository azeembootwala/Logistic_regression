import  numpy as np
import matplotlib.pyplot as plt
import pandas



def get_data(V):
    if V =="train":
        df = pandas.read_csv("train.csv")
        Y = np.asarray(df.ix[:,0])
        X = np.asarray(df.ix[:,1:])
        return X,Y
    else:
        df=pandas.read_csv("test.csv")
        X = np.asarray(df)
        return X

def init_filter(shape, pool_size=(2,2)):
    W = np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1])+shape[-1]*np.prod(shape[:-2]/np.prod(pool_size)))
    return W

def rearrange(X):
    N = X.shape[0]
    out = np.zeros((N,int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])),1),dtype=np.float32)
    for i in range(N):
        out[i,:,:,0] = X[i,:].reshape(int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])))
    return out
def getImageData():
    X, Y = get_data("train")
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

def view_image(number):
    X,Y = get_data("train")
    img = X[number,:]
    img = img.reshape(28,28)
    plt.imshow(img,cmap="gray")
    plt.show()

def tar2ind(Y): # this function converts our targets into indicator matrix
    N = len(Y)
    Y = Y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, Y[i]] = 1
    return ind


def sigmoid(Z):
    Z = np.exp(-Z)
    return 1/(1+Z)
def relu(Z):
    return Z*(Z>0)
def softmax(Z):
    return np.exp(Z)/np.exp(Z).sum(axis = 1,keepdims=True)

def forward(X,activation,W1,b1,W2,b2):
    if activation==1:
        Z =X.dot(W1)+b1
        Z[Z<0]=0
    else:
        Z = np.tanh(X.dot(W1)+b1)

    output = softmax(Z.dot(W2)+b2) #Feed forward function we could try using tanh /relu
    return output,Z

def cross_entropy(ind,output):
    return -(ind*np.log(output)).sum()
def classification_rate(Y,result):
    return np.mean(Y==result)

def derivative_W2(T,Y,Z,reg,W2):
    #return ((T-Y).T.dot(Z)).T
    return Z.T.dot(T-Y)+reg*W2

def derivative_b2(T,Y,reg,b2):
    return (T-Y).sum(axis = 0) + reg*b2

def derivative_W1(T,Y,Z,W2,X,activation,reg,W1):
    if activation==1:
        #dz = (T-Y).dot(W2.T)*Z*(1-Z)
        return X.T.dot( ( ( T-Y ).dot(W2.T) * (Z > 0) ) )
    else:
        dz = (T-Y).dot(W2.T)*(1-Z*Z)
        return X.T.dot(dz)+reg*W1

def derivative_b1(T,Y,Z,W2,activation,reg,b1):
    if activation ==1:
        #dz=((T-Y).dot(W2.T)*Z*(1-Z)).sum(axis = 0)+reg*b1
        return (( T-Y ).dot(W2.T) * (Z > 0)).sum(axis=0)

    else:
        dz=((T-Y).dot(W2.T)*(1-Z*Z)).sum(axis = 0)+reg*b1
        return dz
def init_weight_and_bias(M1,M2):
    W = np.random.randn(M1,M2)/np.sqrt(M1+M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def y2indicator(y):
    N=len(y)
    D=len(set(y))
    ind = np.zeros((N,D),dtype=np.float32)
    for i in range(N):
        ind[i,y[i]]=1
    return ind

def error_rate(a,b):
    return np.mean(a!=b)

if __name__=="__main__":
    main()
