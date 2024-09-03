import numpy as np
from scipy.ndimage import label

from . import matplotlib_utils

# 合法的面积区间
MIN_CIRCLE_AREA = 100
MAX_CIRCLE_AREA = 500
MIN_D = 11
MAX_D = 23
MAX_DIF = 3 # 限制横纵坐标延伸长度差距

# 根据对数分析球的材质
BALL_MATERIAL = 6.7
BALL_MATERIAL_DELTA = 0.7

# 选定某一指定编号，找到对应连通区域，计算坐标集合
def get_coord_list_from_labeled_array(labeled_array, i):
    coordinates = np.argwhere(labeled_array == i) # 获取坐标集合
    x_coords = coordinates[:, 0]  # 行坐标
    y_coords = coordinates[:, 1]  # 列坐标
    return x_coords, y_coords

# 检查获得的形状是否可以近似被认为是圆形区域
def check_approx_circle(x_coords, y_coords): 
    x_range = x_coords.max() - x_coords.min() # 计算极差
    y_range = y_coords.max() - y_coords.min()
    return not(MIN_D <= x_range <= MAX_D and MIN_D <= y_range <= MAX_D and abs(x_range - y_range) <= MAX_DIF)

# 给定一个零一矩阵，假设零是墙壁，一是空洞，返回填后的连通情况，同一连同快
def get_numbered_flood_fill(numpy_array, raw_image):
    labeled_array, ncnt = label(numpy_array)
    check_is_ball = np.zeros(numpy_array.shape) # 增加一个额外掩码，用于进行材质性检查
    check_is_ball[raw_image >= BALL_MATERIAL - BALL_MATERIAL_DELTA] = 1
    check_is_ball[raw_image >= BALL_MATERIAL + BALL_MATERIAL_DELTA] = 0
    labeled_array = labeled_array * check_is_ball # * get_arch() # 2024-09-03 暂时去掉赦免区域
    for i in range(1, ncnt+1):
        if not (MIN_CIRCLE_AREA <= np.sum((labeled_array == i)) <= MAX_CIRCLE_AREA):
            labeled_array[labeled_array == i] = -1 # deleted
        else:
            x_coords, y_coords = get_coord_list_from_labeled_array(labeled_array, i)
            if check_approx_circle(x_coords, y_coords):
                labeled_array[labeled_array == i] = -1 # deleted
            else:
                pass # 合理的连同区域，意味着这个连同区域很可能成为答案
    return labeled_array

# 统计是否找到了圆形
# 其中 numpy_array 必须是一个零一矩阵，0 代表墙，1 代表空地，我们需要对所有连通区域进行 flood_fill
def check_circle_exist(numpy_array, raw_image):
    labeled_array = get_numbered_flood_fill(numpy_array, raw_image)
    unique_elements, counts = np.unique(labeled_array, return_counts=True)
    # 这里为什么以 2：因为一定存在 -1 和 0 两种取值
    # 而有其他取值存在说明找到了额外的符合条件的连通区域
    return len(unique_elements) > 2 

# 将 x 坐标序列，与 y 坐标序列，合并为一个有序对序列
def get_center_position(xcoords, ycoords):
    return (round(xcoords.mean()), round(ycoords.mean()))

# 调用此函数时，需要保证 numpy_array 对应的线框图中确实已经找到了符合近似圆条件的连通区域
# numpy_array 必须是一个零一矩阵，0 代表墙，1 代表空地，我们需要对所有连通区域进行 flood_fill
# 最后返回的结果是一个 list of list
# 内层 list 是近似圆标定连通区域的坐标集合
def get_all_xy_center_coord_list(numpy_array, raw_image) -> list:
    labeled_array = get_numbered_flood_fill(numpy_array, raw_image)
    unique_elements, counts = np.unique(labeled_array, return_counts=True)
    assert len(unique_elements) > 2
    arr = []
    for val, cnt in zip(unique_elements, counts): # 拿来所有有序对
        if val not in [-1, 0]:
            xcoords, ycoords = get_coord_list_from_labeled_array(labeled_array, val)
            assert len(xcoords) == cnt and len(ycoords) == cnt
            arr.append(get_center_position(xcoords, ycoords))
    return arr

# 用于测试
# 能够在检测到圆形结构时输出对比图像
def show_flood_fill(numpy_array, raw_image):
    labeled_array = get_numbered_flood_fill(numpy_array, raw_image)
    if check_circle_exist(numpy_array, raw_image):
        matplotlib_utils.show_three_image_in_one_line(numpy_array, labeled_array, raw_image)
