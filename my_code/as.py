print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Our dataset and targets
# y = (x-10)^2
X = [  [1, 4],
            [3, 4],
            [1, 0.2],
            [3, 4],
            [15, 10], # 异常
            [-10, 4],  # 异常
            [-1, -5],
            [-2, -3],
            [-4, 4],
            [-5, -1.5],
            #------
            [10, 10],
            [5, 5], # 异常
            [2, 3], # 异常
            [12, 5],
            [11, 3.4],
            [-7, -8],
            [-8, -9],
            [-16, -7],
            [-12, -8],
            [-14, -4]
]
Y = [1] * 10 + [0] * 10
kernel = "rbf"
clf = svm.SVC(kernel=kernel, C=0.5, gamma=1)
clf.fit(X, Y)
res = clf.predict([[1, 1], [5, 6], [5, 4]])
print(res)

# # figure number
# fignum = 1
#
# # fit the model
# for kernel in ('linear', 'poly', 'rbf'):
#     clf = svm.SVC(kernel=kernel, gamma=2)
#     clf.fit(X, Y)
#
#     # plot the line, the points, and the nearest vectors to the plane
#     plt.figure(fignum, figsize=(4, 3))
#     plt.clf()
#
#     plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#                 facecolors='none', zorder=10, edgecolors='k')
#     plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
#                 edgecolors='k')
#
#     plt.axis('tight')
#     x_min = -3
#     x_max = 3
#     y_min = -3
#     y_max = 3
#
#     XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
#     Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(XX.shape)
#     plt.figure(fignum, figsize=(4, 3))
#     plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
#     plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#                 levels=[-.5, 0, .5])
#
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
#
#     plt.xticks(())
#     plt.yticks(())
#     fignum = fignum + 1
# plt.show()