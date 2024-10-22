from tqdm import tqdm

from . import advanced_box_method
from . import cube_get
from . import matplotlib_utils
from . import stderr_log
from . import os_interface
from . import dcm_interface
from . import image_log
from . import calibration

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
def svm_check_all_file_in_folder_and_dump_log(folder: str):
    item_list = advanced_box_method.get_all_posible_box_in_folder(folder)
    stderr_log.log_info("dumping images log into log folder.")
    coord_dict = {}
    for i in range(len(item_list)):
        item          = item_list[i]
        timenow       = item["timenow"] # 时间轴坐标
        box_rng       = item["box_rng"] # 记录时间空间坐标
        if coord_dict.get(timenow) is None: # 记录每个时刻所有圈圈的中点坐标
            coord_dict[timenow] = []
        coord_dict[timenow].append(advanced_box_method.get_center_pos_in_box_rng(box_rng))# 计算坐标中点
    for timenow in tqdm(coord_dict):
        filename      = os_interface.get_dcm_filename_by_index(timenow, folder)
        log_numpy_arr = dcm_interface.get_log_numpy_array_from_dcm_file(filename)
        image         = image_log.create_image_from_log_numpy_array_with_center_coord_list(log_numpy_arr, coord_dict[timenow])
        image_log.save_image_to_log_folder(image)

# 根据不同的类别选择颜色模式，以形象地展示识别效果
# 这里的 checker 实际上就是用 advanced_box_method.svm_checker
def select_color_with_checker(checker, image2d):
    tag = checker(image2d)
    if tag == "is_not_ball": # 灰色显示不是球的东西
        return "Grays"
    elif tag == "is_small_ball": # 蓝色显示小球
        return "Blues"
    elif tag == "is_large_ball": # 绿色显示大球
        return "Greens"
    else:
        assert False # tag not found

# 选择一个指定的 36x36x36 的区域
# 保存到日志文件夹
def save_6x6_sample_for_certain_cube(folder, index, center_x, center_y, checker):
    image3d = cube_get.get_cube_from_log_numpy_list_in_folder_around_center(folder, index, center_x, center_y)
    matplotlib_utils.show_6x6_numpy_array(image3d, save=True, show=False, color_selector=lambda image2d: select_color_with_checker(checker, image2d))

# 已经完成了中心点检测的任务后
# 为了方便人去检查中心点位置是否正确，输出一些相关图片到日志文件
def dump_images_for_debug(ball_centers, folder):
    for item in ball_centers:
        time = item["time"]
        xpos = round(item["xpos"])
        ypos = round(item["ypos"])
        filename      = os_interface.get_dcm_filename_by_index(time, folder)
        log_numpy_arr = dcm_interface.get_log_numpy_array_from_dcm_file(filename)
        image         = image_log.create_image_from_log_numpy_array_with_center_coord_list(log_numpy_arr, [(xpos, ypos)])
        image_log.save_image_to_log_folder(image)
        save_6x6_sample_for_certain_cube(folder, time, xpos, ypos, advanced_box_method.svm_checker)
    stderr_log.log_tips("relevant images in: %s" % os_interface.LOG_IMAGE_FOLDER)

# 对检测到的标志物中心的帧进行圈圈处理，并将圈圈后的图片存入日志
# 看起来精度已经很准很准了，不知道后续还是否需要继续优化
# 不要在生产环境中使用此功能
def svm_get_ball_centers_in_folder_and_dump_log(folder: str, debug=True) -> list:
    ball_centers = advanced_box_method.get_all_cluster_center_in_folder(folder)
    stderr_log.log_info("dumping <<<32[%d]>>> images log into log folder." % (len(ball_centers) * 2))
    if debug: # 输出用于调试的图片信息
        dump_images_for_debug(ball_centers, folder)
    return ball_centers # list of dict