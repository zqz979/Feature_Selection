import numpy as np
import matplotlib.pyplot as plt
f_test = np.loadtxt('./F-test.txt').reshape((28,28))
mi = np.loadtxt('./MutualInfo.txt').reshape((28,28))
logistic_regression= np.loadtxt('./LogisticRegression.txt')
logistic_regression_mean= np.loadtxt('./LogisticRegression_mean.txt').reshape((28,28))
logistic_regression_sum = np.loadtxt('./LogisticRegression_sum.txt').reshape((28,28))
mrmr = np.loadtxt('./MRMR.txt').reshape((28,28))

fig = plt.figure(figsize=(10, 10))
rows = 2
columns = 3

fig.add_subplot(rows, columns, 1)
plt.imshow(f_test)
plt.title("F-test")

fig.add_subplot(rows, columns, 2)
plt.imshow(mi)
plt.title("Mutual Information")

fig.add_subplot(rows, columns, 3)
plt.imshow(logistic_regression[0].reshape((28,28)))
plt.title("Logistic Regression")

fig.add_subplot(rows, columns, 4)
plt.imshow(logistic_regression_mean)
plt.title("Logistic Regression with mean")

fig.add_subplot(rows, columns, 5)
plt.imshow(logistic_regression_sum)
plt.title("Logistic Regression with sum")

fig.add_subplot(rows, columns, 6)
plt.imshow(mrmr)
plt.title("mRMR")

plt.show()

