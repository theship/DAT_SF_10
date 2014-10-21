## General Assembly Data Science, San Francisco
## github.com/ga-students/DAT_SF_10
##
## Julee Burdekin
## juburdekin@gmail.com
##
## 20141020
## HW2
##
## Deep, hands-on experience with cross validation and selection of model parameters.
## Although the Scikitlearn package provides packaged methods, crossvalidation is such 
## an important concept that will will implement it ourselves in this assignment.
## We will then use our implementation of cross validation to select some model paraters -
## also called hyperparamters 0 for our KNN classifier on the Iris dataset.

## QUESTION 1: Implement KNN classification, using the sklearn package. We learned how to do this in class.
import numpy as np
import pandas as pd

from sklearn import datasets
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.cross_validation import KFold

feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

iris = pd.read_csv('iris.csv', names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target_names'])

'''
type(iris)
iris.keys()
print iris.info()
'''
print iris.describe()
'''
print iris.keys()
'''

iris = iris.dropna()
X = iris.as_matrix(feature_names).astype(float)
y = iris.as_matrix(['target_names']).astype(str)
y = np.ravel(y)

## split the data into training set and test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=0)

# Train KNN classifier defined function on the train data
from sklearn.neighbors import KNeighborsClassifier
myknn = KNeighborsClassifier(3).fit(X_train,y_train)
knnScore = myknn.score(X_test, y_test)

print knnScore

## QUESTION 2: Implement cross-validation for your KNN classifier.

# generic cross validation function
def cross_validate(X, y, classifier, k_fold):

    # derive a set of (random) training and testing indices
    k_fold_indices = KFold( len(X), n_folds=k_fold,
                           indices=True, shuffle=True,
                           random_state=0)

    k_score_total = 0
    # for each training and testing slices run the classifier, and score the results
    for train_slice, test_slice in k_fold_indices:

        model = classifier(X[ train_slice  ],
                         y[ train_slice  ])

        k_score = model.score(X[ test_slice ],
                              y[ test_slice ])

        k_score_total += k_score

    # return the average accuracy
    averAccuracy =  k_score_total/k_fold

    return averAccuracy

print cross_validate(X, y, KNeighborsClassifier(3).fit, 5)

## QUESTION 3: Use your knn classifier (knnScore) and xvalidation code from Q1&2 (averAccuracy), above
## to get optimal K (# of nearest neighbors to consult)

knn_values = np.array(range(1,120))
knn_results = []
for points in range(1,120):
    knn_results.append(cross_validate(X, y, KNeighborsClassifier(points).fit, 5))

maxK = max(knn_results)
minK = min(knn_results)
meanK = np.array(knn_results).mean()
optimalK_max = knn_results.index(maxK)

print "The top k most similar pieces of data from our known dataset is:"
print optimalK_max
print "But, I am not sure what I'm doing here..."

## QUESTION 4: Use matplotlib to plot classifier accuracy vs. hyperparameter K
## for a range of K that you consider interesting - & explain

plt.plot(knn_results)
plt.show()
