import os
from tqdm import tqdm
from . import dcm_interface
from . import stderr_log
from . import os_interface
from . import image_log
from . import matplotlib_utils

# ------------------------------ 以下内容不得用于生产环境 ------------------------------ #

# 显示一个图像矩阵，每个位置为该位置随时间变化的的极差
# 用于计算静态豁免：即如果 CT 图像中某个像素亮度随时间变化的极差十分小，那么这个位置不可能是标志物
def show_raw_aided_matrix_for_folder(folder: str, thresh: float):
    assert 0 < thresh < 1
    range_matrix = dcm_interface.get_raw_aided_matrix_for_log_znorm_in_folder(folder)
    range_matrix[range_matrix <= thresh] = 0
    matplotlib_utils.show_debug_numpy_array(range_matrix)

# 依次输出每张图片，测试某个文件中的所有 dcm 文件，对数据进行必要的二值化
# 不要在生产环境中使用此功能
def preprocess_all_file_in_folder_and_dump_log(folder: str):
    assert os.path.isdir(folder)
    index_to_coord_set_map = dcm_interface.get_border_based_indexer(folder)
    index_list = []
    for index in index_to_coord_set_map: # 获取所有图像中的识别情况，得到的数据中包含识别出的类似物中心
        index_list.append(index)
    stderr_log.log_info("dumping image log into log folder.")
    for i in tqdm(range(len(index_list))):
        index = index_list[i]
        coord_list = index_to_coord_set_map[index]
        if len(coord_list) > 0: # 绘制给人看的辅助视图，并将辅助视图存储进日志文件夹
            filename      = os_interface.get_dcm_filename_by_index(index, folder)
            log_numpy_arr = dcm_interface.get_log_numpy_array_from_dcm_file(filename)
            image         = image_log.create_image_from_log_numpy_array_with_center_coord_list(log_numpy_arr, coord_list)
            image_log.save_image_to_log_folder(image)
