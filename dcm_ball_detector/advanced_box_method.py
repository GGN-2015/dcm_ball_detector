# 基于 svm 的二次筛选过程
import functools
import numpy as np
from . import svm_utils
from . import image_log
from . import cube_get
from .os_interface import POS_IMAGE, NEG_IMAGE, INNER_POS_IMAGE, INNER_NEG_IMAGE

# 根据预先准备的数据集学习识别标志物的方法
# svm_1 用于区分一般图样和标志物图样
# svm_2 用于区分标志物图样出现时的大与小的不同
@functools.cache
def get_available_svm():
    svm_1 = svm_utils.get_svm_predictor(POS_IMAGE, NEG_IMAGE)
    svm_2 = svm_utils.get_svm_predictor(INNER_POS_IMAGE, INNER_NEG_IMAGE)
    return svm_1, svm_2

# 对 36x36 的 log 对数数据进行预测
def get_prediction_on_log_numpy_array(log_numpy_array):
    svm1, svm2  = get_available_svm()
    image       = image_log.create_image_from_log_numpy_array(log_numpy_array)
    numpy_array = np.array(image)
    assert numpy_array.shape == log_numpy_array.shape
    numpy_array = numpy_array.flatten()
    if svm1.predict([numpy_array])[0] == 1: # 说明外层 svm 呈阴性
        return False
    if svm2.predict([numpy_array])[0] == 1: # 说明内层 svm 呈阴性
        return False
    return True # 说明内层外层都呈现阳性

# 寻找所有可能成为标志物的时刻以及矩形框
# 在先前筛选的基础上，进一步引入 SVM 进行筛选
def get_all_posible_box_in_folder(folder: str) -> list:
    dataset = cube_get.get_all_detected_picture_from_folder(folder)
    arr = []
    for item in dataset:
        timenow = item["timenow"] # 时间轴坐标
        box_rng = item["box_rng"] # 记录时间空间坐标
        image2d = item["image2d"]
        if get_prediction_on_log_numpy_array(image2d):
            arr.append({
                "timenow": timenow,
                "box_rng": box_rng
            })
    return arr