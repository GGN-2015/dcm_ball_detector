from . import cube_get
from . import matplotlib_utils
from . import dcm_interface

# 读取一个长方体区域
def show_data_cube_from_folder(folder: str, tnow: int, xnow: int, ynow: int):
    numpy_arr_3d = cube_get.get_cube_from_log_numpy_list_in_folder_around_center(folder, tnow, xnow, ynow)
    matplotlib_utils.show_6x6_numpy_array(numpy_arr_3d)

# 从一个 CT 数据集中检测所有标志类似物，然后输出到屏幕
def show_all_cube_detected_in_folder(folder):
    index_to_coord_set_map = dcm_interface.get_border_based_indexer(folder)
    index_list = []
    for index in index_to_coord_set_map: # 获取所有图像中的识别情况，得到的数据中包含识别出的类似物中心
        index_list.append(index)
    for i in range(len(index_list)):
        index = index_list[i]
        if len(index_to_coord_set_map[index]) > 0: # 说明能够找到至少一个类似物
            for (center_x, center_y) in index_to_coord_set_map[index]:
                show_data_cube_from_folder(folder, index, center_x, center_y)