import os
import numpy as np
import functools

from . import os_interface
from . import dcm_interface
from . import convolve_utils
from . import gen_circ

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
def get_box_range(tnow: int, xnow: int, ynow: int, max_x_range, max_y_range):
    tmin = tnow - T_RADIUS # 保证时间轴上总长度是完全平方数
    tmax = tnow + T_RADIUS # 以便于将来在水平面上对堆叠的图像进行展平
    xmin = xnow - X_RADIUS 
    xmax = xnow + X_RADIUS
    ymin = ynow - Y_RADIUS
    ymax = ynow + Y_RADIUS
    tmin, tmax = range_unify(tmin, tmax, 0, None)
    xmin, xmax = range_unify(xmin, xmax, 0, max_x_range)
    ymin, ymax = range_unify(ymin, ymax, 0, max_y_range)
    return tmin, tmax, xmin, xmax, ymin, ymax

# 指定中心点的时刻以及坐标，求一个附近的足够大的包围盒
# 我们最好保证 t 坐标上的长度是完全平方数，以便于将来对该维度进行展开
@functools.cache
def get_cube_from_log_numpy_list_in_folder_around_center(folder:str, tnow:int, xnow:int, ynow:int):
    max_x_range, max_y_range = dcm_interface.get_max_xy_range_from_folder(folder)
    tmin, tmax,xmin, xmax, ymin, ymax = get_box_range(tnow, xnow, ynow, max_x_range, max_y_range)
    return get_cube_from_log_numpy_list_in_folder(folder, tmin, tmax, xmin, xmax, ymin, ymax)

# 获取所有可能成为标志物的立方体截图
# 同时记录立方体的角点坐标
def get_all_cube_from_folder(folder:str) -> list:
    index_to_coord_set_map = dcm_interface.get_border_based_indexer(folder)
    max_x_range, max_y_range = dcm_interface.get_max_xy_range_from_folder(folder)
    index_list = []
    for index in index_to_coord_set_map: # 获取所有图像中的识别情况，得到的数据中包含识别出的类似物中心
        index_list.append(index)
    dataset = []
    for i in range(len(index_list)):
        index = index_list[i]
        if len(index_to_coord_set_map[index]) > 0: # 说明能够找到至少一个类似物
            for (center_x, center_y) in index_to_coord_set_map[index]:
                box_rng = get_box_range(index, center_x, center_y, max_x_range, max_y_range)
                image3d = get_cube_from_log_numpy_list_in_folder_around_center(folder, index, center_x, center_y)
                dataset.append({
                    "box_rng": box_rng,
                    "image3d": image3d # numpy array 3d: 36x36x36
                })
    return dataset

# 把所有立方体拆分成独立的图片
def get_all_detected_picture_from_folder(folder: str):
    dataset = get_all_cube_from_folder(folder)
    new_dataset = []
    for item in dataset:
        box_rng = item["box_rng"]
        image3d = item["image3d"]
        tmin, tmax, xmin, xmax, ymin, ymax = box_rng
        for t in range(0, tmax - tmin):
            image2d = image3d[t] # 截取一张图片
            new_dataset.append({
                "timenow": t + tmin,
                "box_rng": (xmin, xmax, ymin, ymax), # 二维意义下的图像坐标
                "image2d": image2d
            })
    return new_dataset

# 把所有立方体拆分成图像，并且按照与标准圆的距离从小到大排序
# 与标准圆距离越小排名越靠前
def get_all_detected_picture_from_folder_and_sort(folder: str):
    dataset       = get_all_detected_picture_from_folder(folder)
    standard_circ = gen_circ.gen_standard_circ()
    return sorted(dataset, key=lambda item: convolve_utils.get_matrix_distance(item["image2d"], standard_circ))