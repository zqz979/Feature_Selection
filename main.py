import numpy as np

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

'''
# F-test
from sklearn.feature_selection import f_classif
f = f_classif(train_data.data.reshape(-1,28*28), train_data.targets)[0]
print("F-test:\n" + str(f))
np.savetxt('F-test.txt', f)
'''

'''
# mutual information
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(train_data.data.reshape(-1,28*28), train_data.targets)
print("Mutual Information:\n" + str(mi))
np.savetxt('MutualInfo.txt', mi)
'''


'''from sklearn.linear_model import SGDClassifier
logreg = SGDClassifier(loss='log',penalty='elasticnet')
logreg.fit(train_data.data.reshape(-1,28*28), train_data.targets)
np.savetxt('LogisticRegression.txt', logreg.coef_)'''

# sum or mean
#print("Logistic Regression:\n" + str(logreg.coef_.sum(axis=0)))
'''np.savetxt('LogisticRegression_sum.txt', logreg.coef_.sum(axis=0))
np.savetxt('LogisticRegression_mean.txt', logreg.coef_.mean(axis=0))'''



'''
import pandas as pd
from sklearn.feature_selection import f_classif

X = pd.DataFrame(train_data.data.reshape(-1,28*28).numpy())
y = pd.DataFrame(train_data.targets.numpy())
def mrmr(X,y,K):
    F = pd.Series(f_classif(X, y)[0], index = X.columns)
    corr = pd.DataFrame(.00001, index = X.columns, columns = X.columns)

    # initialize list of selected features and list of excluded features
    selected = []
    not_selected = X.columns.to_list()

    # repeat K times
    for i in range(K):
    
        # compute (absolute) correlations between the last selected feature and all the (currently) excluded features
        if i > 0:
            last_selected = selected[-1]
            corr.loc[not_selected, last_selected] = X[not_selected].corrwith(X[last_selected]).abs().clip(.00001)
            
        # compute FCQ score for all the (currently) excluded features (this is Formula 2)
        score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis = 1).fillna(.00001)
        
        # find best feature, add it to selected and remove it from not_selected
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)
    return selected

selected = mrmr(X, y, 28*28)
print("MRMR:\n" + str(selected))
np.savetxt('MRMR.txt', selected)
'''