# 基于 svm 的二次筛选过程
import functools
import numpy as np
from . import svm_utils
from . import image_log
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
    svm1, svm2 = get_available_svm()
    image = image_log.create_image_from_log_numpy_array(log_numpy_array)
    numpy_array = np.array(image).flatten()
    assert numpy_array.shape == log_numpy_array.shape
    if svm1.predict([numpy_array])[0] == 1: # 说明外层 svm 呈阴性
        return False
    if svm2.predict([numpy_array])[0] == 1: # 说明内层 svm 呈阴性
        return False
    return True # 说明内层外层都呈现阳性