# 用于将初筛的结果可视化
# 以便于调试程序
# 不得用于生产环境
import numpy as np
import os
from PIL import Image, ImageDraw
from . import os_interface

# 默认的圈出半径
DEFAULT_RADIUS = 15

# log_numpy_array 是函数 get_log_numpy_array_from_dcm_file 的返回值
# 我们需要将这个 log_numpy_array 转化在灰度图
def create_image_from_log_numpy_array(log_numpy_array):
    arr = log_numpy_array.copy()
    min_val = np.min(arr) # 确保数组值在 0 到 1 之间
    max_val = np.max(arr)
    if max_val != min_val:
        scaled_arr = (arr - min_val) / (max_val - min_val)
        gray_image = (scaled_arr * 255).astype(np.uint8) # 将缩放后的数组转换为 0-255 的灰度图
    else:
        gray_image = np.zeros(log_numpy_array.shape).astype(np.uint8) # 考虑特判空白图片
    image = Image.fromarray(gray_image, mode='L') # 使用 Pillow 创建图像
    return image

# 渲染 log_numpy_array 为灰度图
# 并用明显的颜色从中圈出识别到的标志物对象
def create_image_from_log_numpy_array_with_center_coord_list(log_numpy_array, coord_list):
    image = create_image_from_log_numpy_array(log_numpy_array).convert("RGB")
    draw = ImageDraw.Draw(image)
    for (x, y) in coord_list:
        left   = y - DEFAULT_RADIUS
        right  = y + DEFAULT_RADIUS
        top    = x - DEFAULT_RADIUS
        bottom = x + DEFAULT_RADIUS
        draw.ellipse([left, top, right, bottom], outline='red', width=3)
    return image

# 将某个图片存放近日志文件夹
# 编号从 1 开始自动递增
def save_image_to_log_folder(image):
    assert os.path.isdir(os_interface.LOG_IMAGE_FOLDER)
    new_index = len(os.listdir(os_interface.LOG_IMAGE_FOLDER)) + 1                  # 申请一个新的编号
    filename  = os.path.join(os_interface.LOG_IMAGE_FOLDER, "%07d.png" % new_index) # 获得新文件的文件路径
    if isinstance(image, str): # 字符串将会被视为一个外部已有的文件路径
        image = Image.open(image) 
    image.save(filename)