from curses import killchar
import keyword
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Download MNIST dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

algos = ['f', 'mi', 'logreg', 'mrmr']
k = [10,20,30,40,50,60,70,80,90,100,28*28]
accuracy = pd.DataFrame(index = k, columns = algos)

f = np.nan_to_num(np.loadtxt('./F-test.txt'))
mi = np.loadtxt('./MutualInfo.txt')
logistic_regression_mean= np.loadtxt('./LogisticRegression_mean.txt')
mrmr = np.loadtxt('./MRMR.txt')

ranking = [f,mi,logistic_regression_mean,mrmr]

for i in range(4):
  for nfeats in k:    
    feats = np.argsort(ranking[i])[(0-nfeats):]
    clf = SGDClassifier(loss='hinge').fit(
      train_data.data.reshape(-1,28*28)[:,feats], train_data.targets
    )
    accuracy.loc[nfeats, algos[i]] = clf.score(test_data.data.reshape(-1,28*28)[:,feats], test_data.targets)
accuracy.plot(kind='line')
plt.show()