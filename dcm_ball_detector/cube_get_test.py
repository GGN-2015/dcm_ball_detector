from tqdm import tqdm

from . import cube_get
from . import matplotlib_utils
from . import image_log
from . import stderr_log

# 从一个 CT 数据集中检测所有标志类似物，然后输出到屏幕
# 此功能仅用于测试
def show_all_cube_detected_in_folder(folder):
    datalist = cube_get.get_all_cube_from_folder(folder)
    for item in datalist:
        box_rng = item["box_rng"] # 记录时间空间坐标
        image3d = item["image3d"]
        matplotlib_utils.show_6x6_numpy_array(image3d)

# 识别一个 CT 中的所有可能成为标志物的切片
# 然后根据其与标准圆的欧式距离排序，排序后将图片送入图像日志文件夹
def dump_debug_sorted_pictures_to_log_folder(folder: str):
    sorted_item_list = cube_get.get_all_detected_picture_from_folder_and_sort(folder)
    stderr_log.log_info("dumping sorted pictures to image log.")
    for i in tqdm(range(len(sorted_item_list))):
        item    = sorted_item_list[i]
        timenow = item["timenow"] # 时间轴坐标
        box_rng = item["box_rng"] # 记录时间空间坐标
        image2d = item["image2d"]
        pil_img = image_log.create_image_from_log_numpy_array(image2d)
        image_log.save_image_to_log_folder(pil_img) # 保存图片