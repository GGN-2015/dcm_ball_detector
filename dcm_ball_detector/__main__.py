from . import advanced_box_method_test
from . import os_interface
from . import calibration
from . import stderr_log
from . import dcm_interface

import numpy as np
import sys
import os
import json

# 输出坐标相关信息，分析坐标球的相对顺序
# 不要使用这个函数，目前只用于代码参考
def output_coord_relevant_info(ball_centers, dcm_folder):
    if len(ball_centers) == 4:
        stderr_log.log_info("marker set coord3d info in voxel index (ordered by z-index): ") # 输出原始体素编号模式下的坐标信息
        print(ball_centers)
        meta_data_dict             = dcm_interface.get_metadata_from_dcm_folder(dcm_folder)
        transfered_ball_centers    = calibration.space_transfer_all(ball_centers, meta_data_dict)
        coord3d_with_correct_order = calibration.get_best_marker_order(np.array([
            [ 0.0000000000000000,  0.0000000000000000, 0.0000000000000000], # 盆骨扫描件的标志物空间坐标相对位置关系
            [69.3940693200773921, -0.0000000000000036, 0.0000000000000000], 
            [ 9.1217166176732363, 56.4598693992500529, 0.0000000000000001], 
            [72.4396220656959713, 50.7199986380889953, 0.3294267527434576], 
        ]), transfered_ball_centers)
        stderr_log.log_info("marker set coord3d info in natrual 3d coordination (order corrected): ") # 输出毫米为单位的坐标信息，并矫正标志物顺序
        print(coord3d_with_correct_order)

# 这个是旧的测试函数，不要使用这个函数
def old_main(dcm_folder: str):
    os_interface.clear_log_image() # 清空日志文件夹
    ball_centers = advanced_box_method_test.svm_get_ball_centers_in_folder_and_dump_log(dcm_folder)
    output_coord_relevant_info(ball_centers, dcm_folder)

# 先不考虑其他任何功能先，搞一个最简单的版本
def new_main(dcm_folder: str, debug=False):
    os_interface.clear_log_image() # 清空日志文件夹
    ball_centers            = advanced_box_method_test.svm_get_ball_centers_in_folder_and_dump_log(dcm_folder, debug)
    meta_data_dict          = dcm_interface.get_metadata_from_dcm_folder(dcm_folder)
    stderr_log.log_info("meta_data_dict: %s" % json.dumps(meta_data_dict))
    xdir_len_mm             = dcm_interface.get_xdir_len_mm_from_dcm_folder(dcm_folder)
    transfered_ball_centers = calibration.space_transfer_all(ball_centers, xdir_len_mm, meta_data_dict)
    stderr_log.log_info("marker set coord3d info in natrual 3d coordination (order uncorrected): ") # 输出毫米为单位的坐标信息，不关心坐标的相对位置关系
    print(json.dumps(transfered_ball_centers))

# 显示用法
def output_usage():
    stderr_log.log_error("usage: python3 -m dcm_ball_detector         <<<33[<folder path>]>>>")
    stderr_log.log_error("       python3 -m dcm_ball_detector --debug <<<33[<folder path>]>>>\n")

# 匹配并删除一个指定的字符串
# 找到了返回 True, 没找到返回 False
def abstract_param(param_val: str) -> bool: 
    bool_val = param_val in sys.argv
    sys.argv = [v for v in sys.argv if v != param_val]
    return bool_val

argv_debug = abstract_param("--debug")
if len(sys.argv) != 2: # 命令行参数不正确
    output_usage()
    exit(1)

argv_folder_path = sys.argv[1] # 在指定路径进行标志物识别
if not os.path.isdir(argv_folder_path):
    stderr_log.log_error("folder <<<33[%s]>>> does not exist!" % argv_folder_path)
    exit(1)

new_main(argv_folder_path, argv_debug)
# old_main(argv_folder_path)