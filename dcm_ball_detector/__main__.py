from . import advanced_box_method_test
from . import os_interface

# 只有当外置数据文件存在时，才可以进行基于外置数据文件的测试
assert os_interface.check_outer_sample_exist()

os_interface.clear_log_image() # 清空日志文件夹
advanced_box_method_test.svm_get_ball_centers_in_folder_and_dump_log(os_interface.ATTACHED_SAMPLE)