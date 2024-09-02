import os

def dir_file_scan(filepath: str, suffix: str) -> list: # 获取文件夹中的所有具有指定拓展名的简单文件
    assert os.path.isdir(filepath)
    arr = []
    for file in os.listdir(filepath):
        file_full_path = os.path.join(filepath, file)
        if os.path.isfile(file_full_path) and file_full_path.endswith(suffix):
            arr.append(file_full_path)
    assert len(arr) > 0
    return sorted(arr)
