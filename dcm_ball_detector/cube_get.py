import os
import numpy as np
import functools

from . import os_interface
from . import dcm_interface

# 采样盒子在三个方向上各自的半径
T_RADIUS = 18
X_RADIUS = 18
Y_RADIUS = 18

# 在指定文件夹下，获取一个立方体状态的图像，即一个三维的 numpy 数组，其中第一维度是时间，第二维度和第三维是 x 和 y 坐标
# 需要注意的是，在这个例子里不要用 znorm 正则化
# 我们要求 t 从零开始
@functools.cache
def get_cube_from_log_numpy_list_in_folder(folder: str, tmin:int, tmax:int, xmin:int, xmax:int, ymin:int, ymax:int):
    assert os.path.isdir(folder)
    file_list = os_interface.get_all_dcm_file_in_folder(folder)
    arr = []
    for t in range(tmin, tmax):
        assert 0 <= t < len(file_list)
        np_arr = dcm_interface.get_log_numpy_array_from_dcm_file(file_list[t])
        xlen, ylen = np_arr.shape       # 确保确实是二维矩阵
        assert 0 <= xmin < xmax <= xlen # 确保坐标范围合理
        assert 0 <= ymin < ymax <= ylen
        arr.append(np_arr[xmin:xmax, ymin:ymax])
    stacked_array = np.stack(tuple(arr), axis=0)
    assert len(stacked_array.shape) == 3 # 这是一个三维数组
    return stacked_array

# 将采样区域移动到可控范围内
def range_unify(vmin, vmax, l_bound, r_bound):
    if l_bound is not None and vmin < l_bound:
        vmax += (l_bound - vmin)
        vmin = l_bound
    if r_bound is not None and vmax > r_bound:
        vmin -= (vmax - r_bound)
        vmax = r_bound
    return vmin, vmax

# 给定中心点坐标
# 获取采样坐标盒子
def get_box_range(tnow: int, xnow: int, ynow: int):
    tmin = tnow - T_RADIUS # 保证时间轴上总长度是完全平方数
    tmax = tnow + T_RADIUS # 以便于将来在水平面上对堆叠的图像进行展平
    xmin = xnow - X_RADIUS 
    xmax = xnow + X_RADIUS
    ymin = ynow - Y_RADIUS
    ymax = ynow + Y_RADIUS
    tmin, tmax = range_unify(tmin, tmax, 0, None)
    xmin, xmax = range_unify(xmin, xmax, 0, 512)
    ymin, ymax = range_unify(ymin, ymax, 0, 512)
    return tmin, tmax, xmin, xmax, ymin, ymax

# 指定中心点的时刻以及坐标，求一个附近的足够大的包围盒
# 我们最好保证 t 坐标上的长度是完全平方数，以便于将来对该维度进行展开
def get_cube_from_log_numpy_list_in_folder_around_center(folder:str, tnow:int, xnow:int, ynow:int):
    tmin, tmax,xmin, xmax, ymin, ymax = get_box_range(tnow, xnow, ynow)
    return get_cube_from_log_numpy_list_in_folder(folder, tmin, tmax, xmin, xmax, ymin, ymax)
