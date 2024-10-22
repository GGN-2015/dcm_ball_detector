import random
import string
import numpy as np

from . import os_interface
from . import image_log

# 由于我们只会在测试环境使用 matplotlib 因此不要直接引入它
# 我们只在 show_debug_numpy_array 函数中使用了这个包
try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except:
    has_matplotlib = False

# 输出拟合的二次函数图像信息
def debug_output_curve(x_data, y_data, a, b, c):
    x_fit = np.linspace(13, 23, 100) # 生成拟合曲线的 y 值
    y_fit = a * x_fit**2 + b * x_fit + c
    plt.scatter(x_data, y_data, color='red', label='Data Points') # 可视化原始点和拟合曲线
    plt.plot(x_fit, y_fit, color='blue', label='Fitted Quadratic Curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Quadratic Curve Fitting')
    plt.show()

# 输出测试
def debug_output_for_connected_component(connected_component, hull, centroid):
    plt.plot(connected_component[:, 1], connected_component[:, 0], 'o', markersize=8) # 可视化凸包
    for simplex in hull.simplices:
        plt.plot(connected_component[simplex, 1], connected_component[simplex, 0], 'k-') # 画出凸包
    plt.plot(centroid[0], centroid[1], 'r*', markersize=15, label='Centroid')
    plt.gca().invert_yaxis()  # 翻转 y 轴以匹配图像坐标系
    plt.show()

# 仅用于测试，不要用于生产环境
def show_debug_numpy_array(numpy_array, title="2darray"):
    if has_matplotlib:
        plt.imshow(numpy_array, cmap='gray') # 显示灰度影像
        plt.title(title)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

# 并列显示三个灰度图
# 用于展示三个 numpy array
def show_three_image_in_one_line(numpy_array_1, numpy_array_2, numpy_array_3):
    plt.figure(figsize=(17, 9))
    plt.subplot(1, 3, 1)
    plt.title('Image A')
    plt.imshow(numpy_array_1, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Image B')
    plt.imshow(numpy_array_2, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Image CS')
    plt.imshow(numpy_array_3, cmap='gray')
    plt.show()

# 随机选择字符并生成字符串
def generate_random_string(length):
    characters = string.ascii_letters + string.digits + '_' + '-' # 定义字符集，包括字母、数字和下划线
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# 给定一个 36xPxQ 的矩阵
# 把他以 6x6 的图片形式输出到屏幕上
# color_selector 用于根据具体图片选择合适的颜色模式以供图片的展示
def show_6x6_numpy_array(single_numpy_array, save=False, show=True, color_selector=None):
    assert len(single_numpy_array.shape) == 3
    tlen, xlen, ylen = single_numpy_array.shape
    assert tlen == 36 # 不支持其他值的处理
    fig, axs = plt.subplots(6, 6, figsize=(15, 15)) # 设置图像堆叠方式
    for i in range(6):
        for j in range(6):
            if color_selector is None:
                axs[i, j].imshow(single_numpy_array[i * 6 + j], cmap="Grays")
            else:
                axs[i, j].imshow(single_numpy_array[i * 6 + j], cmap=color_selector(single_numpy_array[i * 6 + j]))
    plt.tight_layout() # 调整子图间距
    if show: # 是否输出到屏幕
        plt.show()
    if save: # 是否保存到日志
        try:
            tmp_file_name = "dcm_ball_detector_%s.png" % generate_random_string(64)
            plt.savefig(tmp_file_name)
            image_log.save_image_to_log_folder(tmp_file_name)
        finally:
            os_interface.remove_file(tmp_file_name)