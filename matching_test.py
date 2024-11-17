from scipy.optimize import linear_sum_assignment
import numpy as np
import random
import copy
import time


def gener_matrix(m, n):
    matrix = np.zeros([m, n])
    for i in range(0, m):
        for j in range(0, n):
            matrix[i][j] = random.randint(1, 30)
    return matrix


def my_solution(matrix, m, n, x):
    r = 0
    row_degree = np.zeros(m)
    col_degree = np.zeros(n)
    for k1 in range(0, m * n):
        index = np.argmax(matrix)
        row1 = int(index / n) + 0
        col1 = int(index % n) + 0
        if row_degree[row1] == x or col_degree[col1] == 1:
            matrix[row1][col1] = 0
            continue
        else:
            r += matrix[row1][col1]
            matrix[row1][col1] = 0
            row_degree[row1] += 1
            col_degree[col1] += 1
    return r


def sci_solution(matrix, m, n, x):
    result = 0
    for n1 in range(0, x):
        row, col = linear_sum_assignment(matrix, True)
        for i1 in range(0, len(row)):
            result += matrix[row[i1]][col[i1]]
            for j1 in range(0, m):
                matrix[j1][col[i1]] = 0
    return result


A1 = gener_matrix(64, 1000)
A2 = copy.deepcopy(A1)

start = time.time()
print("原方案score:", my_solution(A1, 64, 1000, 15))
mid = time.time()
print("最大匹配score", sci_solution(A2, 64, 1000, 15))
end = time.time()
print(mid - start, end - mid)
