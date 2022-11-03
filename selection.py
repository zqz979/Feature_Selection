import numpy as np
from sklearn.feature_selection import r_regression
from rdc import *
import matplotlib.pyplot as plt

# Download dataset
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms


transform1 = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])
train_data = datasets.CIFAR10(
    root = 'data',
    train = True,                         
    transform = transform1, 
    download = True,            
)
test_data = datasets.CIFAR10(
    root = 'data', 
    train = False, 
    transform = transform1
)

# show sample RGB separation
'''fig = plt.figure(figsize=(10, 10))
rows = 1
columns = 3
img = train_data.data[0]

fig.add_subplot(rows, columns, 1)
plt.imshow(img[:,:,0],cmap='Reds')
plt.title("R")

fig.add_subplot(rows, columns, 2)
plt.imshow(img[:,:,1],cmap='Greens')
plt.title("G")

fig.add_subplot(rows, columns, 3)
plt.imshow(img[:,:,2],cmap='Blues')
plt.title("B")

plt.show()
'''

#random
np.savetxt('random2.txt', np.random.rand(32*32,1))
print('Random Done!')

'''
channels = ['R', 'G', 'B']
for i in range(1,3):
    # F-test
    from sklearn.feature_selection import f_classif
    f = f_classif(train_data.data[:,:,:,i].reshape(50000,32*32), train_data.targets)[0]
    print("F-Test channel " + channels[i] + " Done!")
    np.savetxt('F-test' + channels[i] + '.txt', f)


    # mutual information
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(train_data.data[:,:,:,i].reshape(50000,32*32), train_data.targets)
    print("Mutual Information " + channels[i] + " Done!")
    np.savetxt('MutualInfo' + channels[i] + '.txt', mi)



    from sklearn.linear_model import SGDClassifier
    logreg = SGDClassifier(loss='log',penalty='elasticnet')
    logreg.fit(train_data.data[:,:,:,i].reshape(50000,32*32), train_data.targets)
    np.savetxt('LogisticRegression' + channels[i] + '.txt', logreg.coef_)

    # sum or mean
    print("Logistic Regression " + channels[i] + " Done!")
    np.savetxt('LogisticRegression_sum' + channels[i] + '.txt', logreg.coef_.sum(axis=0))
    np.savetxt('LogisticRegression_mean' + channels[i] + '.txt', logreg.coef_.mean(axis=0))

    
    rdc = rdc(train_data.data[:,:,:,i].reshape(50000,32*32), train_data.targets, 32*32)
    mrmr = np.divide(f,rdc)
    print('mRMR Done!')
    np.savetxt('mRMR' + channels[i] + '.txt', mrmr)


'''

