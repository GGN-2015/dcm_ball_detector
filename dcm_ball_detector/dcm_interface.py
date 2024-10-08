import os
import numpy as np
import pydicom
from tqdm import tqdm
import functools

from . import flood_fill
from . import stderr_log

# 用于调用操作系统相关的接口
from . import os_interface
from . import matrix_man

AIDED_THRESH        = 0.43
MATCH_CIRC_THRESH   = 80

from .flood_fill import BALL_MATERIAL, BALL_MATERIAL_DELTA

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
# 为了防止额外的空洞产生，我们在此处只对材质 < BALL_MATERIAL - BALL_MATERIAL_DELTA 的位置进行了去除
# 而没有对材质大于 BALL_MATERIAL + BALL_MATERIAL_DELTA 的位置进行去除
@functools.cache
def get_log_numpy_array_from_dcm_file(filepath: str):
    assert os.path.isfile(filepath)
    log_numpy_array = np.log(get_raw_numpy_array_from_dcm_file(filepath)[0] + 1)
    log_numpy_array[log_numpy_array < BALL_MATERIAL - BALL_MATERIAL_DELTA] = 0
    return log_numpy_array

# 读入一个 dcm 文件，返回一个先对数化，再 Z 正则化的 numpy array 数据
def get_log_znorm_numpy_array_from_dcm_file(filepath: str, min_val: float, max_val: float):
    assert os.path.isfile(filepath)
    old_numpy_array = get_log_numpy_array_from_dcm_file(filepath)
    new_numpy_array = (old_numpy_array - min_val) / (max_val - min_val)
    return new_numpy_array

# 给定一个文件夹，读取文件夹中的所有 dcm 文件
# 然后获取这些 dcm 文件中的所有 numpy array 中的 min 和 max 值
@functools.cache
def get_min_max_value_of_log_dataset_folder(folder: str):
    assert os.path.isdir(folder)
    file_list = os_interface.get_all_dcm_file_in_folder(folder) # 被扫描的文件集合
    min_list  = []
    max_list  = []
    stderr_log.log_info("preprocessing log numpy array.")
    for index in tqdm(range(len(file_list))):
        file    = file_list[index]
        min_now = (np.amin(get_log_numpy_array_from_dcm_file(file)))
        max_now = (np.amax(get_log_numpy_array_from_dcm_file(file)))
        min_list.append(min_now)
        max_list.append(max_now)
    return min(min_list), max(max_list)

# 获得某个文件夹中的 dcm 文件中的图片尺寸
def get_max_xy_range_from_folder(folder):
    filename  = os_interface.get_dcm_filename_by_index(0, folder) # 假设所有文件具有相同的尺寸
    log_numpy = get_log_numpy_array_from_dcm_file(filename)
    return log_numpy.shape

# 扫描一个文件夹下的所有文件，进行批量化的对数化与正则化
# 返回得到的全部 numpy array 构成的 list
@functools.cache
def get_all_log_znorm_numpy_array_in_folder(folder: str) -> list:
    assert os.path.isdir(folder)
    min_val, max_val = get_min_max_value_of_log_dataset_folder(folder)
    file_list = os_interface.get_all_dcm_file_in_folder(folder) # 被扫描的文件集合
    arr = []
    for file in file_list:
        arr.append(get_log_znorm_numpy_array_from_dcm_file(file, min_val, max_val))
    return arr

# 对文件夹中正则化后的数据获得极差图样
# 极差图样作为辅助序列用于消除不动区域
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

# 文件夹中有 dcm 序列，输出所有检测到标志球的时刻以及当时时刻标志球的坐标集合
# 此处还没有进行任何基于相对位置与形状的筛选
# 还需要进一步进行聚类、基于模式识别进行筛选
def get_border_based_indexer(folder: str) -> dict:
    assert os.path.isdir(folder)
    min_val, max_val = get_min_max_value_of_log_dataset_folder(folder)
    aided_matrix = get_raw_aided_matrix_for_log_znorm_in_folder(folder)
    aided_matrix[aided_matrix <= AIDED_THRESH] = 0
    # aided_matrix *= arc_mask.get_arch() # 2024-09-03 暂时去掉赦免区域
    dic = {}
    fileset = os_interface.get_all_dcm_file_in_folder(folder)
    stderr_log.log_info("generating border based index.")
    for index in tqdm(range(len(fileset))): # 在每张图片中进行初步筛选，index 从零开始递增
        dic[index] = []
        file = fileset[index]
        np_array = get_log_znorm_numpy_array_from_dcm_file(file, min_val, max_val)
        np_array = matrix_man.get_aided_neighbout_var(np_array, aided_matrix)
        if flood_fill.check_circle_exist(1 - np_array, get_log_numpy_array_from_dcm_file(file)): # 找到了指定连通区域
            dic[index] = flood_fill.get_all_xy_center_coord_list(1 - np_array, get_log_numpy_array_from_dcm_file(file))
    return dic
