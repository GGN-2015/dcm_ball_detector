import numpy as np
from scipy.ndimage import convolve

# 计算矩阵卷积
# 小矩阵是卷积核
def matrix_convolve(big_matrix, small_matrix):
    small_matrix_flipped = np.flipud(np.fliplr(small_matrix))
    result_matrix = convolve(big_matrix, small_matrix_flipped, mode='constant', cval=0.0)
    return result_matrix