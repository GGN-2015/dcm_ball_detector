import os
import functools
import shutil
from . import stderr_log

# 当前脚本所在目录
DIRNOW           = os.path.dirname(os.path.abspath(__file__))
LOG_IMAGE_FOLDER = os.path.join(DIRNOW, "log_image")
POS_IMAGE        = os.path.join(DIRNOW, "ground_truth", "pos")
NEG_IMAGE        = os.path.join(DIRNOW, "ground_truth", "neg")
INNER_POS_IMAGE  = os.path.join(DIRNOW, "ground_truth", "inner_pos", "pos")
INNER_NEG_IMAGE  = os.path.join(DIRNOW, "ground_truth", "inner_pos", "neg")
ROOT_DIR         = os.path.dirname(DIRNOW)
ATTACHED_SAMPLE  = os.path.join(ROOT_DIR, "data_sample", "SE3")

# 删除普通文件
def remove_file(filepath: str):
    if os.path.isfile(filepath): # 如果文件存在
        os.remove(filepath)

# 检查外置测试数据是否存在
def check_outer_sample_exist() -> bool:
    return os.path.isdir(ATTACHED_SAMPLE)

# 保证当前脚本所在目录下有一个空文件夹名为 log_image
def clear_log_image():
    if os.path.isdir(LOG_IMAGE_FOLDER):
        shutil.rmtree(LOG_IMAGE_FOLDER)
    os.mkdir(LOG_IMAGE_FOLDER) # 重新创建这个文件夹

# 获取文件夹中的所有具有指定拓展名的简单文件
# 不含目录
@functools.cache
def dir_file_scan(filepath: str, suffix: str) -> list: 
    filepath = os.path.abspath(filepath)
    if not os.path.isdir(filepath):
        stderr_log.log_error("folder <<<33[%s]>>> not found." % filepath)
        exit(1)
    arr = []
    for file in os.listdir(filepath):
        file_full_path = os.path.join(filepath, file)
        if os.path.isfile(file_full_path) and file_full_path.endswith(suffix):
            arr.append(file_full_path)
    if len(arr) == 0:
        stderr_log.log_error("no dcm file found in folder <<<33[%s]>>>." % filepath)
        exit(1)
    return sorted(arr)

# 获取一个文件夹中的全部 dcm 文件
@functools.cache
def get_all_dcm_file_in_folder(folder: str) -> list:
    return dir_file_scan(folder, ".dcm")

# 获取一个文件夹中的所有 dcm 文件并按照文件名字典序排序
# 然后从中取某个出来
@functools.cache
def get_dcm_filename_by_index(index, folder) -> str:
    files = get_all_dcm_file_in_folder(folder)
    return files[index]