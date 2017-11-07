from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas

class LogisticModel(object):
    def __init__(self):
        pass
    def get_data(self,V):
        if V=="train":
            df = pandas.read_csv("train.csv")
            T = np.asarray(df["Survived"]).reshape(df["Survived"].shape[0],1)
        else:
            df = pandas.read_csv("test.csv")

         # Our target variable
        df["Pclass_1"]=df["Pclass"].apply(lambda x : 1 if x==1 else 0)#One hot encoding for our data which had 3 categories
        df["Pclass_2"]=df["Pclass"].apply(lambda x : 1 if x==2 else 0)
        df["Pclass_3"]=df["Pclass"].apply(lambda x : 1 if x==3 else 0)
        df["Family_size"]=df["SibSp"]+df["Parch"]
        del df["Pclass"] #Taking out the colums as we no longer need it
        del df["SibSp"]
        del df["Parch"]
        del df["Cabin"]
        count=df["Name"].shape[0]
        for name,i in zip(df["Name"],range(0,count)):
            if 'Mr' in name:
                df.ix[i,"Title"]="Mr"
            elif "Mrs" in name:
                df.ix[i,"Title"]="Mrs"
            elif "Miss" in name:
                df.ix[i,"Title"]="Miss"
            elif "Master" in name:
                df.ix[i,"Title"]="Master"
            else:
                g=df.ix[i,"Sex"]
                if g=="male":
                    df.ix[i,"Title"]="Mr"
                else:
                    df.ix[i,"Title"]="Mrs"
        df["Title_1"]=df["Title"].apply(lambda x : 1 if x=="Mr" else 0)#One hot encoding for our data which had 3 categories
        df["Title_2"]=df["Title"].apply(lambda x : 1 if x=="Mrs" else 0)
        df["Title_3"]=df["Title"].apply(lambda x : 1 if x=="Master" else 0)
        df["Title_4"]=df["Title"].apply(lambda x : 1 if x=="Miss" else 0)
        del df["Title"]
        del df["Ticket"]
        df["Sex"]=df["Sex"].apply(lambda x : 1 if x=="male" else 0)
        del df["Name"]
        df["Age"]=df["Age"].apply(lambda x : x if x<100 else df["Age"].median())
        df["Fare"]=df["Fare"].apply(lambda x : x if x<100 else df["Fare"].median())
        df["Emb_1"]=df["Embarked"].apply(lambda x : 1 if x=="C" else 0)
        df["Emb_2"]=df["Embarked"].apply(lambda x : 1 if x=="Q" else 0)
        df["Emb_3"]=df["Embarked"].apply(lambda x : 1 if x=="S" else 0)
        del df["Embarked"]
        df["Age"]=df["Age"]/df["Age"].max()
        df["Fare"]=df["Fare"]/df["Fare"].max()
        df["Family_size"]=df["Family_size"]/df["Family_size"].max()
        #df["Age"]=(df["Age"]-df["Age"].mean())/df["Age"].std()
        #df["Fare"]=(df["Fare"]-df["Fare"].mean())/df["Fare"].std()
        #df["Family_size"]=(df["Family_size"]-df["Family_size"].mean())/df["Family_size"].std()
        X = np.asarray(df.ix[:,"Sex":])
        if V =="train":
            return X,T
        else:
            return X

    def fit_model(self,X,T,lr =10e-4,reg=10e-1,epoch=3000,show_fig=False):
        N,D = X.shape
        self.W = np.random.randn(D,2)#change1 2 dimentional weight
        X,T = shuffle(X,T)

        T_mat = self.target2_ind(T) #change 5
        cost =[]
        for i in range(0,epoch):
            Y_pred = self.forward(X)

            c=self.cross_entropy(T_mat,Y_pred)

            self.W = self.W - lr*(X.T.dot(Y_pred-T_mat)+reg*np.sign(self.W))


            if i%40 ==0:
                Yhat = np.argmax(Y_pred,axis=1) #change 4 added this line
                Yhat=Yhat.reshape(Yhat.shape[0],1)
                r = self.classification_rate(Yhat,np.array(T))
                cost.append(c)
                print("i: " ,i,"cost: ",c,"rate:", r )
        if show_fig:
            plt.plot(cost)
            plt.show()
        return self.W



    def soft_max(self,Z):
        Z = np.exp(Z)
        return Z/Z.sum(axis = 1,keepdims=True)#change2 sigmoid to soft_max
    def forward(self,X):
        out = self.soft_max(X.dot(self.W))
        return out
    def cross_entropy(self,T,Y):
        return -(T*np.log(Y)).sum() #change3  the cross_entropy cost
    def classification_rate(self,Y,T):
        #Y=np.round(Y)
        return np.mean(Y==T)
    def target2_ind(self,T):#change 6
        N = len(T)
        D = 2
        T_mat = np.zeros((N,D))
        for i in range(0,N):
            num = int(T[i])
            T_mat[i,num]=1
        return T_mat



def main():
    model = LogisticModel()
    X_train,Y_train = model.get_data("train") # WE loaded the training data
    # we have a class imbalance problem , we will add 100 more datasets to the survived category
    X_class1=[]
    count = 0
    for i in range(0,Y_train.shape[0]):
        if Y_train[i]==1:
            X_class1.append(X_train[i,:])
            count+=1
        if count==100:
            break
    X_class1 = np.asarray(X_class1)
    X = np.vstack([X_train,X_class1])
    Y_class1 = np.ones((100,1))
    T = np.vstack([Y_train,Y_class1])

    #Adding a bias term
    bias = np.ones((X.shape[0],1))
    X = np.hstack([X,bias])


    W=model.fit_model(X,T,show_fig=True)
    X_test= model.get_data("test")
    bias1 = np.ones((X_test.shape[0],1))
    X_test = np.hstack([X_test,bias1])
    Yhat = model.forward(X_test)
    Yhat = np.argmax(Yhat,axis=1).reshape(Yhat.shape[0],1) #Yhat =np.round(Yhat)


    df1 = pandas.read_csv("test.csv")
    Submit = np.asarray(df1.ix[:,"PassengerId"]).reshape(418,1)


    result = np.hstack([Submit,Yhat])
    df_final =pandas.DataFrame({"PassengerID":result[:,0],"Survived":result[:,1]},dtype=int)
    df_final["Survived"]=df_final["Survived"].apply(lambda x: 0 if x==None else x)
    df_final.to_csv("submission.csv",sep=",",index=False,dtype=int)

if __name__ == '__main__':
    main()
