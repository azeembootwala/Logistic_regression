
# coding: utf-8

# In[9]:

import os
import numpy as np
import pandas


# In[10]:

def get_data(V):
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


# In[11]:
