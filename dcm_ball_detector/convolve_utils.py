import numpy as np
from scipy.ndimage import convolve

# 计算矩阵卷积
# 小矩阵是卷积核
def matrix_convolve(big_matrix, small_matrix):
    small_matrix_flipped = np.flipud(np.fliplr(small_matrix))
    result_matrix = convolve(big_matrix, small_matrix_flipped, mode='constant', cval=0.0)
    return result_matrix

# 用于计算两个矩阵之间的欧式距离
def get_matrix_distance(matrix1, matrix2) -> float:
    assert matrix1.shape == matrix2.shape
    distance = np.sqrt(np.sum((matrix1 - matrix2) ** 2))
    return float(distance)