import os
import numpy as np
import pydicom
from scipy.ndimage import uniform_filter
import functools

from . import circ_utils
from . import convolve_utils
from . import flood_fill
from . import arc_mask

# 用于调用操作系统相关的接口
from . import os_interface

DEFAULT_WINDOW_SIZE = 3
VAR_THRESH          = 0.043
AIDED_THRESH        = 0.43
MATCH_CIRC_THRESH   = 80

from .flood_fill import BALL_MATERIAL, BALL_MATERIAL_DELTA

def get_neighbour_var(numpy_array): # 计算邻域方差矩阵并二值化
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

# 在不变性协助的前提下，考虑使用卷积滤波识别矩阵中的小圆
# 矩阵中指定位置的值越大，约有可能是发生了匹配
def get_matched_aided_neighbout_var(numpy_array, aided_matrix):
    aided_matrix = get_aided_neighbout_var(numpy_array, aided_matrix)
    circ_matrix  = circ_utils.get_circle_mask()
    conv_matrix  = convolve_utils.convolve(aided_matrix, circ_matrix)
    conv_matrix[conv_matrix <= MATCH_CIRC_THRESH] = 0
    return conv_matrix

# ------------------------------ 以下内容与 dcm 文件读取有关 ------------------------------ #

# 读入一个 dcm 文件，获取未经正则化的原生的 numpy array 数据
# 返回一个二元组，第一个元素是这个 numpy array，第二个元素是 dataset 元数据信息
@functools.cache
def get_raw_numpy_array_from_dcm_file(filepath: str):
    assert os.path.isfile(filepath)
    dataset = pydicom.dcmread(filepath)
    return (dataset.pixel_array.copy(), dataset)

# 读入一个 dcm 文件，返回一个对数化后的 numpy array 数据
# 这里根据材质进行了一次筛选
def get_log_numpy_array_from_dcm_file(filepath: str):
    assert os.path.isfile(filepath)
    log_numpy_array = np.log(get_raw_numpy_array_from_dcm_file(filepath)[0] + 1)
    log_numpy_array[log_numpy_array < BALL_MATERIAL - BALL_MATERIAL_DELTA] = 0
    # log_numpy_array[log_numpy_array > BALL_MATERIAL + BALL_MATERIAL_DELTA] = 0
    return log_numpy_array

# 读入一个 dcm 文件，返回一个先对数化，再 Z 正则化的 numpy array 数据
def get_log_znorm_numpy_array_from_dcm_file(filepath: str, min_val: float, max_val: float):
    assert os.path.isfile(filepath)
    old_numpy_array = get_log_numpy_array_from_dcm_file(filepath)
    new_numpy_array = (old_numpy_array - min_val) / (max_val - min_val)
    return new_numpy_array

# 对 znorm 的结果进行二值化
# def get_log_znorm_bin_numpy_array_from_dcm_file(filepath: str, min_val: float, max_val: float, thresh: float):
#     assert os.path.isfile(filepath)
#     assert 0 < thresh < 1
#     numpy_array = get_log_znorm_numpy_array_from_dcm_file(filepath, min_val, max_val).copy()
#     numpy_array[numpy_array <= thresh] = 0
#     numpy_array[numpy_array  > thresh] = 1
#     return numpy_array

# 给定一个文件夹，读取文件夹中的所有 dcm 文件
# 然后获取这些 dcm 文件中的所有 numpy array 中的 min 和 max 值
def get_min_max_value_of_log_dataset_folder(folder: str):
    assert os.path.isdir(folder)
    file_list = os_interface.dir_file_scan(folder, ".dcm") # 被扫描的文件集合
    min_list  = []
    max_list  = []
    for file in file_list:
        min_now = (np.amin(get_log_numpy_array_from_dcm_file(file)))
        max_now = (np.amax(get_log_numpy_array_from_dcm_file(file)))
        min_list.append(min_now)
        max_list.append(max_now)
    return min(min_list), max(max_list)

# 扫描一个文件夹下的所有文件，进行批量化的对数化与正则化
# 返回得到的全部 numpy array 构成的 list
@functools.cache
def get_all_log_znorm_numpy_array_in_folder(folder: str) -> list:
    assert os.path.isdir(folder)
    min_val, max_val = get_min_max_value_of_log_dataset_folder(folder)
    file_list = os_interface.dir_file_scan(folder, ".dcm") # 被扫描的文件集合
    arr = []
    for file in file_list:
        arr.append(get_log_znorm_numpy_array_from_dcm_file(file, min_val, max_val))
    return arr

# 对文件夹中正则化后的数据获得极差图样
@functools.cache
def get_raw_aided_matrix_for_log_znorm_in_folder(folder: str):
    assert os.path.isdir(folder)
    arr = get_all_log_znorm_numpy_array_in_folder(folder)
    max_matrix = arr[0].copy()
    min_matrix = arr[0].copy()
    for i in range(1, len(arr)):
        max_matrix = np.maximum(max_matrix, arr[i])
        min_matrix = np.minimum(min_matrix, arr[i])
    return max_matrix - min_matrix

# ------------------------------ 以下内容不得用于生产环境 ------------------------------ #

# 由于我们只会在测试环境使用 matplotlib 因此不要直接引入它
# 我们只在 show_debug_numpy_array 函数中使用了这个包
try:
    import matplotlib.pyplot as plt
except:
    pass

# 显示一个 512x512 的矩阵，每个位置为该位置随时间变化的的极差
def show_raw_aided_matrix_for_folder(folder: str, thresh: float):
    assert 0 < thresh < 1
    range_matrix = get_raw_aided_matrix_for_log_znorm_in_folder(folder)
    range_matrix[range_matrix <= thresh] = 0
    show_debug_numpy_array(range_matrix)

# 仅用于测试，不要用于生产环境
def show_debug_numpy_array(numpy_array):
    plt.imshow(numpy_array, cmap='gray') # 显示灰度影像
    plt.axis('off')  # 隐藏坐标轴
    plt.show()

# 仅用于测试，测试 DCM 文件的读取是否正常
# 不要在生产环境中使用此功能
def show_debug_dcm_file(dcm_file_path: str):
    assert os.path.isfile(dcm_file_path) # 检查文件是否存在
    dataset = pydicom.dcmread(dcm_file_path)

    min_value   = np.min(dataset.pixel_array) # 保证正值
    pixel_array = dataset.pixel_array.copy() - min_value
    pixel_array = np.log(pixel_array + 1)

    new_min_value = np.min(pixel_array) # Z 正则化
    new_max_value = np.max(pixel_array)
    pixel_array   = (pixel_array - new_min_value) / (new_max_value - new_min_value)
    show_debug_numpy_array(pixel_array)

# 依次输出每张图片，测试某个文件中的所有 dcm 文件，对数据进行必要的二值化
# 不要在生产环境中使用此功能
def show_debug_all_file_in_folder(folder: str):
    assert os.path.isdir(folder)
    show_debug_numpy_array(arc_mask.get_arch())
    min_val, max_val = get_min_max_value_of_log_dataset_folder(folder)
    aided_matrix = get_raw_aided_matrix_for_log_znorm_in_folder(folder)
    aided_matrix[aided_matrix <= AIDED_THRESH] = 0
    aided_matrix *= arc_mask.get_arch()
    show_debug_numpy_array(aided_matrix)
    for index, file in enumerate(os_interface.dir_file_scan(folder, ".dcm")):
        np_array = get_log_znorm_numpy_array_from_dcm_file(file, min_val, max_val)
        np_array = get_aided_neighbout_var(np_array, aided_matrix)
        print(index)
        flood_fill.show_flood_fill(1 - np_array, get_log_numpy_array_from_dcm_file(file))
