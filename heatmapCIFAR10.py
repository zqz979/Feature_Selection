import numpy as np
import matplotlib.pyplot as plt

f_test_r = np.loadtxt('./F-testR.txt').reshape((32,32))
f_test_g = np.loadtxt('./F-testG.txt').reshape((32,32))
f_test_b = np.loadtxt('./F-testB.txt').reshape((32,32))

mi_r = np.loadtxt('./MutualInfoR.txt').reshape((32,32))
mi_g = np.loadtxt('./MutualInfoG.txt').reshape((32,32))
mi_b = np.loadtxt('./MutualInfoB.txt').reshape((32,32))

logistic_regression_mean_r= np.loadtxt('./LogisticRegression_meanR.txt').reshape((32,32))
logistic_regression_mean_g= np.loadtxt('./LogisticRegression_meanG.txt').reshape((32,32))
logistic_regression_mean_b= np.loadtxt('./LogisticRegression_meanB.txt').reshape((32,32))

logistic_regression_sum_r = np.loadtxt('./LogisticRegression_sumR.txt').reshape((32,32))
logistic_regression_sum_g = np.loadtxt('./LogisticRegression_sumG.txt').reshape((32,32))
logistic_regression_sum_b = np.loadtxt('./LogisticRegression_sumB.txt').reshape((32,32))

'''mrmr = np.loadtxt('./mRMR.txt').reshape((28,28))
'''
rd = np.loadtxt('./random.txt').reshape((32,32))


fig = plt.figure(figsize=(10, 10))
rows = 5
columns = 3


fig.add_subplot(rows, columns, 1)
plt.imshow(f_test_r)
plt.title("F-test R")

fig.add_subplot(rows, columns, 2)
plt.imshow(f_test_g)
plt.title("F-test G")

fig.add_subplot(rows, columns, 3)
plt.imshow(f_test_b)
plt.title("F-test B")


fig.add_subplot(rows, columns, 4)
plt.imshow(mi_r)
plt.title("Mutual Information R")

fig.add_subplot(rows, columns, 5)
plt.imshow(mi_g)
plt.title("Mutual Information G")

fig.add_subplot(rows, columns, 6)
plt.imshow(mi_b)
plt.title("Mutual Information B")


fig.add_subplot(rows, columns, 7)
plt.imshow(logistic_regression_mean_r)
plt.title("Logistic Regression with mean R")

fig.add_subplot(rows, columns, 8)
plt.imshow(logistic_regression_mean_g)
plt.title("Logistic Regression with mean G")

fig.add_subplot(rows, columns, 9)
plt.imshow(logistic_regression_mean_b)
plt.title("Logistic Regression with mean B")


fig.add_subplot(rows, columns, 10)
plt.imshow(logistic_regression_sum_r)
plt.title("Logistic Regression with sum R")

fig.add_subplot(rows, columns, 11)
plt.imshow(logistic_regression_sum_g)
plt.title("Logistic Regression with sum G")

fig.add_subplot(rows, columns, 12)
plt.imshow(logistic_regression_sum_b)
plt.title("Logistic Regression with sum B")

'''fig.add_subplot(rows, columns, 6)
plt.imshow(mrmr)
plt.title("mRMR")'''

fig.add_subplot(rows, columns, 13)
plt.imshow(rd)
plt.title("random")

plt.show()

