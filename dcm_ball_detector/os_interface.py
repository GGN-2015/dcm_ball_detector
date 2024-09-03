import os
import functools
import shutil

# 当前脚本所在目录
DIRNOW           = os.path.dirname(os.path.abspath(__file__))
LOG_IMAGE_FOLDER = os.path.join(DIRNOW, "log_image")
POS_IMAGE        = os.path.join(DIRNOW, "ground_truth", "pos")
NEG_IMAGE        = os.path.join(DIRNOW, "ground_truth", "neg")
INNER_POS_IMAGE  = os.path.join(DIRNOW, "ground_truth", "inner_pos", "pos")
INNER_NEG_IMAGE  = os.path.join(DIRNOW, "ground_truth", "inner_pos", "neg")

# 保证当前脚本所在目录下有一个空文件夹名为 log_image
def clear_log_image():
    if os.path.isdir(LOG_IMAGE_FOLDER):
        shutil.rmtree(LOG_IMAGE_FOLDER)
    os.mkdir(LOG_IMAGE_FOLDER) # 重新创建这个文件夹

# 获取文件夹中的所有具有指定拓展名的简单文件
# 不含目录
@functools.cache
def dir_file_scan(filepath: str, suffix: str) -> list: 
    assert os.path.isdir(filepath)
    arr = []
    for file in os.listdir(filepath):
        file_full_path = os.path.join(filepath, file)
        if os.path.isfile(file_full_path) and file_full_path.endswith(suffix):
            arr.append(file_full_path)
    assert len(arr) > 0
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