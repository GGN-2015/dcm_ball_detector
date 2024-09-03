import numpy as np
import functools
from .flood_fill import BALL_MATERIAL

# 在 36x36 的坐标系下获得一个 r=14 的标准的圆
@functools.cache
def gen_standard_circ():
    array_size = 36
    circle_array = np.zeros((array_size, array_size))
    center = (array_size // 2, array_size // 2)
    radius = 14
    for i in range(array_size):
        for j in range(array_size):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                circle_array[i, j] = BALL_MATERIAL  # 将圆形区域设置为 BALL_MATERIAL
    return circle_array