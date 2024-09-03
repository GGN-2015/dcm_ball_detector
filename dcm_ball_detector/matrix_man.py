import numpy as np
from scipy.ndimage import uniform_filter

# 基于方差的边界识别与二值化处理
DEFAULT_WINDOW_SIZE = 3
VAR_THRESH          = 0.043

# 计算邻域方差矩阵并二值化
# 借此实现边界识别
def get_neighbour_var(numpy_array): 
    assert len(numpy_array.shape) == 2
    mean         = uniform_filter(numpy_array     , size=DEFAULT_WINDOW_SIZE)
    mean_squared = uniform_filter(numpy_array ** 2, size=DEFAULT_WINDOW_SIZE)
    varirans_mat = mean_squared - mean ** 2
    min_val = np.min(varirans_mat)
    max_val = np.max(varirans_mat)
    varirans_mat = (varirans_mat - min_val) / (max_val - min_val)
    varirans_mat[varirans_mat <= VAR_THRESH] = 0
    varirans_mat[varirans_mat  > VAR_THRESH] = 1
    return varirans_mat

# aided_matrix 是一个形状相同的矩阵，该矩阵中位置为零的位置应该在 numpy_array 的方差矩阵中也被视为零
def get_aided_neighbout_var(numpy_array, aided_matrix):
    var_numpy_array = get_neighbour_var(numpy_array)
    var_numpy_array[aided_matrix <= 0] = 0
    return var_numpy_array
