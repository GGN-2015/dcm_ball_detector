import numpy as np
from .arc_mask import get_arch

try:
    import matplotlib.pyplot as plt
except:
    pass

from scipy.ndimage import label

# 合法的面积区间
MIN_CIRCLE_AREA = 100
MAX_CIRCLE_AREA = 400
MIN_D = 11
MAX_D = 23
MAX_DIF = 3 # 限制横纵坐标延伸长度差距

# 根据对数分析球的材质
BALL_MATERIAL = 6.7
BALL_MATERIAL_DELTA = 0.7

# 给定一个零一矩阵，假设零是墙壁，一是空洞，返回填后的连通情况，同一连同快
def get_numbered_flood_fill(numpy_array, raw_image):
    labeled_array, ncnt = label(numpy_array)
    check_is_ball = np.zeros(numpy_array.shape) # 增加一个额外掩码
    check_is_ball[raw_image >= BALL_MATERIAL - BALL_MATERIAL_DELTA] = 1
    check_is_ball[raw_image >= BALL_MATERIAL + BALL_MATERIAL_DELTA] = 0
    labeled_array = labeled_array * check_is_ball * get_arch()
    for i in range(1, ncnt+1):
        if not (MIN_CIRCLE_AREA <= np.sum((labeled_array == i)) <= MAX_CIRCLE_AREA):
            labeled_array[labeled_array == i] = -1 # deleted
        else:
            coordinates = np.argwhere(labeled_array == i) # 获取坐标集合
            x_coords = coordinates[:, 0]  # 行坐标
            y_coords = coordinates[:, 1]  # 列坐标
            x_range = x_coords.max() - x_coords.min() # 计算极差
            y_range = y_coords.max() - y_coords.min()
            print(i, x_range, y_range)
            if not(MIN_D <= x_range <= MAX_D and MIN_D <= y_range <= MAX_D and abs(x_range - y_range) <= MAX_DIF):
                labeled_array[labeled_array == i] = -1 # deleted
            else:
                print(i, np.sum((labeled_array == i)))
    return labeled_array

# 统计是否找到了圆形
def check_circle_exist(numpy_array, raw_image):
    labeled_array = get_numbered_flood_fill(numpy_array, raw_image)
    unique_elements, counts = np.unique(labeled_array, return_counts=True)
    return len(unique_elements) > 2

# 用于测试
def show_flood_fill(numpy_array, raw_image):
    labeled_array = get_numbered_flood_fill(numpy_array, raw_image)
    if check_circle_exist(numpy_array, raw_image):
        plt.figure(figsize=(17, 9))
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(numpy_array, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('Labeled Image')
        plt.imshow(labeled_array, cmap='nipy_spectral')
        plt.subplot(1, 3, 3)
        plt.title('Raw Image')
        plt.imshow(raw_image, cmap='gray')
        plt.show()
