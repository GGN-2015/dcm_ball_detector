
# 由于我们只会在测试环境使用 matplotlib 因此不要直接引入它
# 我们只在 show_debug_numpy_array 函数中使用了这个包
try:
    import matplotlib.pyplot as plt
except:
    pass

# 仅用于测试，不要用于生产环境
def show_debug_numpy_array(numpy_array):
    plt.imshow(numpy_array, cmap='gray') # 显示灰度影像
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

# 给定一个 36xPxQ 的矩阵
# 把他以 6x6 的图片形式输出到屏幕上
def show_6x6_numpy_array(single_numpy_array):
    assert len(single_numpy_array.shape) == 3
    tlen, xlen,ylen = single_numpy_array.shape
    assert tlen == 36 # 不支持其他值的处理
    fig, axs = plt.subplots(6, 6, figsize=(15, 15)) # 设置图像堆叠方式
    for i in range(6):
        for j in range(6):
            axs[i, j].imshow(single_numpy_array[i * 6 + j], cmap="gray")  # 绘制随机数据
    plt.tight_layout() # 调整子图间距
    plt.show()