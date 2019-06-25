#!/usr/bin/env python
# @Time    : 2019/6/22 18:18
# @Author  : Swift  
# @File    : qwe.py
# @Brief   : 
# @Link    :

import numpy as np
import matplotlib.pyplot as plt

# X = np.c_[  (1, 4),
#             (3, 4),
#             (1, 0.2),
#             (3, 4),
#             (15, 10), # 异常
#             (-10, 4),  # 异常
#             (-1, -5),
#             (-2, -3),
#             (-4, 4),
#             (-5, -1.5),
#             #------
#             (10, 10),
#             (5, 5),
#             (2, 3), # 异常
#             (12, 5),
#             (11, 3.4),
#             (-7, -5.5),
#             (-8, -3.8),
#             (-6, -7),
#             (-6.5, -8),
#             (-14, -4)].T

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
            [5, 5],
            [2, 3], # 异常
            [12, 5],
            [11, 3.4],
            [-7, -5.5],
            [-8, -3.8],
            [-6, -7],
            [-6.5, -8],
            [-14, -4]
]


Y = [1] * 10 + [0] * 10

fig, ax = plt.subplots()
for i in range(20):
    if i < 10:
        ax.scatter(X[i][0], X[i][1], c="green")
    else:
        ax.scatter(X[i][0], X[i][1], c="red")
plt.show()


# x, y = np.random.rand(2, 10)
# x = [12, 45, 36]
# y = [10, 43, 15]
#
# fig, ax = plt.subplots()
#
# for i in range(3):
#     if i%2 == 0:
#         ax.scatter(x[i:i+1], y[i:i+1], c="green")
#     else:
#         ax.scatter(x[i:i + 1], y[i:i + 1], c="red")
#
# plt.show()