from tqdm import tqdm

from . import advanced_box_method
from . import cube_get
from . import matplotlib_utils
from . import stderr_log
from . import os_interface
from . import dcm_interface
from . import image_log

# 用于测试：展示识别到的正确的标志物区域形态
def show_all_posible_box_in_folder(folder: str):
    dataset = cube_get.get_all_detected_picture_from_folder(folder)
    for item in dataset:
        timenow = item["timenow"] # 时间轴坐标
        box_rng = item["box_rng"] # 记录时间空间坐标
        image2d = item["image2d"]
        if advanced_box_method.get_prediction_on_log_numpy_array(image2d):
            matplotlib_utils.show_debug_numpy_array(image2d)

# 对所有检测到标志物的帧进行圈圈处理，并将圈圈后的图片存入日志
# 不要在生产环境中使用此功能
def svm_check_all_file_in_folder_and_dump_log(folder: str):
    item_list = advanced_box_method.get_all_posible_box_in_folder(folder)
    stderr_log.log_info("dcm_ball_detector: dumping image log into log folder.")
    coord_dict = {}
    for i in range(len(item_list)):
        item          = item_list[i]
        timenow       = item["timenow"] # 时间轴坐标
        box_rng       = item["box_rng"] # 记录时间空间坐标
        if coord_dict.get(timenow) is None: # 记录每个时刻所有圈圈的中点坐标
            coord_dict[timenow] = []
        coord_dict[timenow].append((round((box_rng[0] + box_rng[1])/2), round((box_rng[2] + box_rng[3])/2)))# 计算坐标中点
    for timenow in tqdm(coord_dict):
        filename      = os_interface.get_dcm_filename_by_index(timenow, folder)
        log_numpy_arr = dcm_interface.get_log_numpy_array_from_dcm_file(filename)
        image         = image_log.create_image_from_log_numpy_array_with_center_coord_list(log_numpy_arr, coord_dict[timenow])
        image_log.save_image_to_log_folder(image)