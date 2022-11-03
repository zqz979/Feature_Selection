import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Download MNIST dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.CIFAR10(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.CIFAR10(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)
  
algos = ['f', 'mi', 'logreg','random']
k = [100,200,300,400,500,600,700,800,900,32*32]
accuracy = pd.DataFrame(index = k, columns = algos)

f = np.nan_to_num(np.loadtxt('./F-testR.txt'))
mi = np.loadtxt('./MutualInfoR.txt')
logistic_regression_mean= np.loadtxt('./LogisticRegression_meanR.txt')
random_importance = np.loadtxt('./random2.txt')


ranking = [f,mi,logistic_regression_mean,random_importance]

for i in range(4):
  for nfeats in k:
    feats = np.argsort(ranking[i])[(0-nfeats):]
    clf = KNeighborsClassifier(n_neighbors=12).fit(
      train_data.data[:,:,:,0].reshape(50000,32*32)[:,feats], train_data.targets
    )
    accuracy.loc[nfeats, algos[i]] = clf.score(test_data.data[:,:,:,0].reshape(10000,32*32)[:,feats], test_data.targets)
    print(str(nfeats) + ' in ' + algos[i] + ' Done!')
  print(algos[i] + 'Done!')
accuracy.plot(kind='line')
plt.show()
