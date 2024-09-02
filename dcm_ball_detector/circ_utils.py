import numpy as np
import functools
import matplotlib.pyplot as plt

# 对圆内杂乱进行惩罚的力度
DEFAULT_NEG_VALUE = -0.3

# 获得小圆匹配性掩码
@functools.cache
def get_circle_mask():
    array_size = 25
    image  = np.ones((array_size, array_size))
    image *= DEFAULT_NEG_VALUE

    # 定义圆的参数
    center = (12, 12)   # 圆心坐标
    radius = 11         # 半径
    line_width = 3      # 线宽

    # 绘制圆
    for i in range(array_size):
        for j in range(array_size):
            # 计算当前点到圆心的距离
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            # 如果距离在圆的边界范围内，则设置为 1
            if radius - line_width / 2 <= distance <= radius + line_width / 2:
                image[i, j] = 1  # 或者设置为其他值，表示圆的颜色
    return image

if __name__ == "__main__":
    # 显示结果
    plt.imshow(get_circle_mask(), cmap='gray')
    plt.axis('off')  # 隐藏坐标轴
    plt.show()