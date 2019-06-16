import numpy as np
def getbinary():
    first=True
    Y=[]
    X=[]
    for line in open("fer2013.csv","r"):
        row = line.split(",")
        if first:
            first=False
        else:
            y=int(row[0])
            if y==0 or y==1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.asarray(X)/255.0, np.asarray(Y)
