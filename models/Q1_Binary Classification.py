#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# In[1]:


import os
import numpy as np
from numpy import load
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import xml.etree.ElementTree as et

import skimage.io
import skimage.color
from skimage import img_as_float
from skimage.io import imread_collection,imread
from skimage.color import rgb2gray
from skimage.transform import resize

import decimal
decimal.getcontext().prec = 100


# # Data Loading and Preprocessing

# In[2]:


data = load('pneumoniamnist.npz')
lst = data.files
xtrain = data[lst[0]]
xval = data[lst[1]]
xtest = data[lst[2]]
ytrain = data[lst[3]]
yval = data[lst[4]]
ytest = data[lst[5]]

xtrain = np.reshape(xtrain,(-1,28*28))
xval = np.reshape(xval,(-1,28*28))
xtest = np.reshape(xtest,(-1,28*28))


# In[3]:


class OneHotEncoder:
    def __init__(self,classes=2,type='none'):
        self.classes = classes
        self.type = type
        pass

    def transform(self,X:np.ndarray):
        self.classes = int(np.max(X)+1)
        if self.type=='none':
            Y = -1*np.ones((len(X),self.classes))
        else:
            Y = np.zeros((len(X),self.classes))

        for i in range(len(X)):
            Y[i,X[i]] = 1
        return Y

    def inverse_transform(self,X:np.ndarray):
        Y = np.zeros(len(X))
        for i in range(len(X)):
            Y[i] = 0
            mx = X[i,0]
            for j in range(self.classes):
                if X[i,j] > mx:
                    Y[i] = j
                    mx = X[i,j] 
        return Y

class StandardScaler:
    def __init__(self):
        pass
  
    def fit(self,X:np.ndarray):
        self.mean = np.mean(X)
        self.std = np.std(X)
  
    def transform(self,X:np.ndarray):
        return (X-self.mean)/self.std

    def inverse_transform(self,X:np.ndarray):
        return X*self.std+self.mean


# In[4]:


x_scalar = StandardScaler()
x_scalar.fit(xtrain)
xtrain = x_scalar.transform(xtrain)
xval = x_scalar.transform(xval)
xtest = x_scalar.transform(xtest)

#Binary Classification
encoder = OneHotEncoder(2,type='zeros') 

ytrain = encoder.transform(ytrain)


# #  Performance Metric Related Functions

# In[5]:


def MeanSquaredError(ydata,ypredict):
    return np.sqrt(np.mean(np.square(ydata-ypredict)))

def ReturnConfMatrix(ypredict,ytest,N=2):
    Conf = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            a = np.where(ypredict==i,1,0)
            b = np.where(ytest==j,1,0)
            c = a*b
            Conf[i,j] = np.sum(c)
    return Conf

def EvaluateAccuracy(ypredict,ytest,N=2):
    Conf = ReturnConfMatrix(ypredict,ytest,N)
    I = np.identity(N)
    Diag = Conf*I
    return np.sum(Diag)/max(np.sum(Conf),0.001)

def EvaluatePrecision(ypredict,ytest,i,N=2):
    Conf = ReturnConfMatrix(ypredict,ytest,N)
    return max(Conf[i,i],0.001)/max(np.sum(Conf[i,:]),0.001)

def EvaluateRecall(ypredict,ytest,i,N=2):
    Conf = ReturnConfMatrix(ypredict,ytest,N)
    return max(Conf[i,i],0.001)/max(np.sum(Conf[:,i]),0.001)

def EvaluateF1score(ypredict,ytest,i,N=2):
    p = EvaluatePrecision(ypredict,ytest,i,N)
    r = EvaluateRecall(ypredict,ytest,i,N)
    return 2*r*p/(r+p)

def EvaluateFalsePositiveRate(ypredict,ytest,i,N=2):
    Conf = ReturnConfMatrix(ypredict,ytest,N)
    t1=max(np.sum(Conf[i,:]),0.001)
    t2=t1-Conf[i,i]
    t3=max(t2,0.001)
    return t3/max(np.sum(Conf[i,:]),0.001)

def EvaluateAUC(ypredict,ytest,i,N=2):
    return EvaluatePrecision(ypredict,ytest,i,N)/EvaluateFalsePositiveRate(ypredict,ytest,i,N)

def EvaluateCategoricalCrossEntropy(ypredict,ytest):
    return np.abs(np.sum(np.log(ypredict)@ytest.transpose()))

def tptnfpfn(ypredict,ytest,N=2):  # Function to calculate True/False Positives/Negatives
    Conf = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            a = np.where(ypredict==i,1,0)
            b = np.where(ytest==j,1,0)
            c = a*b
            Conf[i,j] = np.sum(c)
    tp=np.zeros(N)
    tn=np.zeros(N)
    fp=np.zeros(N)
    fn=np.zeros(N)
    for i in range(N):
        tp[i]=Conf[i,i]
    ConfSum=np.sum(Conf)
    for i in range(N):
        for j in range(N):
            fp[i]+=Conf[j,i]
            fn[i]+=Conf[i,j]
        temp1=fp[i]
        temp2=fn[i]
        fp[i]-=Conf[i,i]
        fn[i]-=Conf[i,i]
        tn[i]=ConfSum-temp1-temp2+Conf[i,i]

    return tp,tn,fp,fn 

def plot(errors,iter,title=''):
    plt.title(title)
    plt.plot(range(iter),errors)
    plt.xticks(np.linspace(start=0,stop=iter,num=11))
    plt.xlabel('Iterations')
    plt.ylabel('MeanSquaredError')
    plt.show()
    plt.clf()
    
def print_performance_metrics(y_predict, y_test, N):
    num = len(y_predict)
    y_predict=np.reshape(y_predict,(num,1))
    y_test=np.reshape(y_test,(num,1))
    
    print('Accuracy',EvaluateAccuracy(y_predict,y_test,N))
    
    for i in range(N):
        f1 = EvaluatePrecision(y_predict,y_test,i,N)
        print(f'Precision : {i} = {f1:.3f}')

    for i in range(N):
        f1 = EvaluateRecall(y_predict,y_test,i,N)
        print(f'Recall : {i} = {f1:.3f}')

    for i in range(N):
        f1 = EvaluateF1score(y_predict,y_test,i,N)
        print(f'F1 Score : {i} = {f1:.3f}')

    for i in range(N):
        f1 = EvaluateAUC(y_predict,y_test,i,N)
        print(f'AUC : {i} = {f1:.3f}')

    print('Confusion Matrix',ReturnConfMatrix(y_predict,y_test,N))

    tp,tn,fp,fn = tptnfpfn(y_predict,y_test,N)
    print('True Positives',tp)
    print('True Negatives',tn)
    print('False Positives',fp)
    print('False Negatives',fn)


# # MLE

# In[6]:


class GaussianMaxLikelihoodEstimate:
    def __init__(self):
        pass

    def fit(self,X:np.ndarray):
        self.mean = np.mean(X,axis=0)
        self.cov = np.cov(X.transpose())
        self.cov = self.cov + 0.01*np.identity(self.cov.shape[0])
        self.det = np.linalg.det(self.cov)
        self.inv = np.linalg.inv(self.cov)
        pass

    def predict(self,x):
        x = x-self.mean
        return np.exp(-1*(x@(self.inv)@x)/2)

class GaussianBayesClassifier:
    def __init__(self):
        pass
    def fit(self,X:np.ndarray,Y:np.ndarray,validation_data=False):
        self.classes = 2
        self.Gmles = []
        for j in range(self.classes):
            data = []
            for x,y in zip(X,Y):
                if y[j]==1:
                    data.append(x)
            data = np.array(data)
            gmle = GaussianMaxLikelihoodEstimate()
            gmle.fit(data)
            self.Gmles.append(gmle)
    
        if type(validation_data)!=bool:
            return EvaluateAccuracy(self.batchPrediction(validation_data[0]),validation_data[1],self.classes)
  
    def predict(self,x):
        curr=-1
        mx=-1
        mx = 0
        for i,gmle in enumerate(self.Gmles):
            res = gmle.predict(x)
            if res > curr:
                mx = i
                curr = res
        return mx


    def batchPrediction(self,X:np.ndarray):
        ypredict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            ypredict[i] = self.predict(X[i,:])
        return ypredict


# In[7]:


gm = GaussianBayesClassifier()
gm.fit(xtrain,ytrain,validation_data=(xval,yval))

ypredict_gm = gm.batchPrediction(xtest)

print_performance_metrics(ypredict_gm, ytest, 2)


# # Logistic Regression

# In[8]:


class LogisticRegressor:
    
    def __init__(self):
        pass

    def preprocess(self,x):
        d = len(x)
        x_new = np.ones(d+1)
        k=0
        for i in x:
            x_new[k] = i
            k+=1
        return x_new
  
    def batchPreprocessing(self,X):
        d = X.shape[1]
        X_new = np.zeros((X.shape[0],d+1))
        for i in range(X.shape[0]):
            X_new[i] = self.preprocess(X[i])
        return X_new
    
    def fit(self,X:np.ndarray,Y:np.ndarray,iter=20,alpha=1,beta=0.9,gamma=0.01,validation_data=False,regularization='none'):
        '''
        X : Input batch
        Y : Output batch
        iter : Iterations
        alpha : step-size
        beta : step-size decay
        gamma : regularization parameter
        regularization : {'none','L1','L2','Elastic'}
        '''
        X = self.batchPreprocessing(X)
        if validation_data!=False:
            xval = self.batchPreprocessing(validation_data[0])

        self.W = np.zeros((Y.shape[1],X.shape[1]))
        Y = np.reshape(Y,(len(Y),-1))
      
        errors = []
        for i in range(iter):
            for j in range(Y.shape[1]):
                if(regularization=='none'):
                    self.W[j,:] = self.W[j,:] - alpha*((self.batchPrediction(X)*(1-self.batchPrediction(X)))[:,j]).transpose()@X/Y.shape[0]
                elif(regularization=='L2'):
                    self.W[j,:] = self.W[j,:] - alpha*(((self.batchPrediction(X)*(1-self.batchPrediction(X)))[:,j]).transpose()@X/Y.shape[0] - gamma*self.W[j,:])
                elif(regularization=='L1'):
                    self.W[j,:] = self.W[j,:] - alpha*(((self.batchPrediction(X)*(1-self.batchPrediction(X)))[:,j]).transpose()@X/Y.shape[0] - gamma*np.sign(self.W[j,:]))
                elif(regularization=='Elastic'):
                    self.W[j,:] = self.W[j,:] - alpha*(((self.batchPrediction(X)*(1-self.batchPrediction(X)))[:,j]).transpose()@X/Y.shape[0] - gamma*np.sign(self.W[j,:]))
        alpha = beta*alpha

        if type(validation_data)!=bool:
            errors.append(EvaluateCategoricalCrossEntropy(self.batchPrediction(xval),validation_data[1]))

        return errors


    def predict(self,x:np.ndarray):
        if x.shape[0]!=self.W.shape[1]:
            x = self.preprocess(x)
        x = np.reshape(x,(len(x),-1))
        y = self.W@x
        for i in range(y.shape[0]):
            y[i] = 1/(1+np.exp(y[i]))
        y = np.reshape(y,(y.shape[0],))
        return y

    def batchPrediction(self,X:np.ndarray):
        ypredict = []
        for i in range(X.shape[0]):
            ypredict.append(self.predict(X[i,:]))
        ypredict = np.array(ypredict)
        return ypredict


# In[9]:


lm = LogisticRegressor()

iter = 50
errors = lm.fit(xtrain,ytrain,alpha=0.001,beta=1,iter=iter,validation_data=(xval,encoder.transform(yval)))


# In[12]:


ypredict_lm = lm.batchPrediction(xtest)
encoder = OneHotEncoder()
ypredict_lm = encoder.inverse_transform(ypredict_lm)

print_performance_metrics(ypredict_lm, ytest, 2)


# # K Nearest Neighbour

# In[13]:


class KNNclassifier:
    def __init__(self):
        pass
    def EUC_DIST(self,v1,v2): 
        v1,v2 = np.array(v1),np.array(v2)
        distance = np.sum((v1-v2)**2)
        return np.sqrt(distance)
    
    def Predict(self,k,Xtrain,Ytrain,Xtest_instance): 
        distances = [] 
        for i in range(len(Xtrain)):
            dist = self.EUC_DIST(Xtrain[i], Xtest_instance)
            distances.append((Ytrain[i],dist)) 
        distances.sort(key=lambda x: x[1]) 
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][0])
        classes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][1]
            if response in classes.keys():
                classes[int(response)] += 1
            else:
                classes[int(response)] = 1
        sorted_classes = sorted(classes.items() , key = lambda x: x[1],reverse = True )
        return sorted_classes[0][0]

    def batchPrediction(self,Knn,Xtrain,Ytrain,Xtest):
        ypredict=[]
        for i in range(Xtest.shape[0]):
            ypredict.append(self.Predict(Knn,Xtrain,Ytrain,Xtest[i,:]))
        ypredict = np.array(ypredict)
        return ypredict


# In[14]:


knn = KNNclassifier()

ypredict_k1 = knn.batchPrediction(1,xtrain,ytrain,xtest)
print_performance_metrics(ypredict_k1, ytest, 2)


# In[15]:


ypredict_k2 = knn.batchPrediction(2,xtrain,ytrain,xtest)
print_performance_metrics(ypredict_k2, ytest, 2)


# In[16]:


ypredict_k5 = knn.batchPrediction(5,xtrain,ytrain,xtest)
print_performance_metrics(ypredict_k5, ytest, 2)


# In[17]:


ypredict_k9 = knn.batchPrediction(9,xtrain,ytrain,xtest)
print_performance_metrics(ypredict_k9, ytest, 2)


# In[18]:


ypredict_k11 = knn.batchPrediction(11,xtrain,ytrain,xtest)
print_performance_metrics(ypredict_k11, ytest, 2)


# # Naive Bayes

# In[19]:


class GaussianNaiveMLE:

    def __init__(self):
        pass
  
    def fit(self,X:np.ndarray):
        self.mean = np.mean(X,axis=0)
        self.cov = np.cov(X.transpose())    
        self.cov = self.cov*np.identity(self.cov.shape[0])
        self.cov = self.cov + 0.01*np.identity(self.cov.shape[0])
        self.det = np.linalg.det(self.cov)
        self.inv = np.linalg.inv(self.cov)
        pass
  
    def predict(self,x):
        x = x-self.mean
        return np.exp(-1*(x@(self.inv)@x)/2)

class NaiveBayesClassifier:
    def __init__(self):
        pass
  
    def fit(self,X:np.ndarray,Y:np.ndarray,validation_data=False):
        x,y = (np.shape(Y))
        self.classes = y
        self.Gmles = []
        for j in range(self.classes):
            data = []
        for x,y in zip(X,Y):
            if y[j]==1:
                data.append(x)
        data = np.array(data)
        gmle = GaussianNaiveMLE()
        gmle.fit(data)
        self.Gmles.append(gmle)
    
        if type(validation_data)!=bool:
            return EvaluateAccuracy(self.batchPrediction(validation_data[0]),validation_data[1],self.classes)
  
    def predict(self,x):
        curr=-1
        mx=-1
        mx = 0
        for i,gmle in enumerate(self.Gmles):
            res = gmle.predict(x)
            if res > curr:
                mx = i
                curr = res
        return mx


    def batchPrediction(self,X:np.ndarray):
        ypredict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            ypredict[i] = self.predict(X[i,:])
        return ypredict


# In[20]:


nbc = NaiveBayesClassifier()
nbc.fit(xtrain,ytrain,validation_data=(xval,yval))
ypredict_nbc = nbc.batchPrediction(xtest)

print_performance_metrics(ypredict_nbc, ytest, 2)


# # Parzen Window

# In[21]:


class ParzenWindow():
    """
    Parzen Window
    """
    def __init__(self, X, window_function="gaussian"):
        self.X = X
    def func_val_gaussian(self, x):
        val = 0.0
        for pts in self.X:
            val += np.exp(-0.5 * np.dot(x-pts, (x-pts).T)) / len(self.X)*(np.sqrt(2 * np.pi))**pts.shape[0]
        return val
    def posterior(self, x):
        _posterior = self.func_val_gaussian(x)
        return _posterior


# In[25]:


def Parzen_pred(xtrain, ytrain, xtest):
    X_1 = []
    X_0 = []
    for i in range(len(ytrain)):
        if ytrain[i][0]==1:
            X_1.append(xtrain[i])
        else:
            X_0.append(xtrain[i])
    X_0 = np.array(X_0)
    X_1 = np.array(X_1)
    post_1 = ParzenWindow(X_1)
    post_0 = ParzenWindow(X_0)
    
    y_pred = np.zeros((624,1))
    for j in range (624):
        if post_1.posterior(xtest[j])>post_0.posterior(xtest[j]):
            y_pred[j] = 1
        else:
            y_pred[j] = 0 
    return y_pred


# In[26]:


y_predict_pw = Parzen_pred(xtrain, ytrain, xtest)


# In[27]:


print_performance_metrics(y_predict_pw, ytest, 2)


# In[ ]:




