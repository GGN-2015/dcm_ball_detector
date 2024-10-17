from . import advanced_box_method_test
from . import os_interface
from . import calibration
from . import stderr_log
from . import dcm_interface
import numpy as np

# 只有当外置数据文件存在时，才可以进行基于外置数据文件的测试
assert os_interface.check_outer_sample_exist()

# 输出坐标相关信息，分析坐标球的相对顺序
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

def main(dcm_folder: str):
    os_interface.clear_log_image() # 清空日志文件夹
    ball_centers = advanced_box_method_test.svm_get_ball_centers_in_folder_and_dump_log(dcm_folder)
    output_coord_relevant_info(ball_centers, dcm_folder)

# 在 repo 自带的样例中进行测试
main(os_interface.ATTACHED_SAMPLE)