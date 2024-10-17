from .advanced_box_method      import get_all_cluster_center_in_folder, get_all_marker_center_and_give_the_mm_coord_with_correct_order
from .advanced_box_method_test import svm_get_ball_centers_in_folder_and_dump_log
from .dcm_relevant_test        import preprocess_all_file_in_folder_and_dump_log
from . import os_interface

# 根据所有的 dcm 影像获取识别到的所有标志物的坐标
__all__ = [
    "get_all_cluster_center_in_folder",                               # 不带图片输出
    "svm_get_ball_centers_in_folder_and_dump_log",                    # 带图片输出
    "preprocess_all_file_in_folder_and_dump_log" ,                    # 仅用于测试
    "get_all_marker_center_and_give_the_mm_coord_with_correct_order", # 输出以毫米为单位的坐标，并调节顺序
]

os_interface.clear_log_image() # 清空日志文件夹