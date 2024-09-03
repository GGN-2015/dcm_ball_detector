from . import dcm_interface

# 别急，还没好好改，仅供测试
# 我劝你们不要看到绝对路径就骂人，我现在原型还没写好呢！！！！
FOLDER1 = "/run/media/neko/Archive_001/Archive/2024-09-02 DogCT/20240117/XH-001-2_20240117-172005-1919_172033/0.625 x 0.625_301"
FOLDER2 = "/run/media/neko/Archive_001/Archive/2024-09-02 DogCT/20240103/XH-1-0103_20240103-145148-1892_145219/0.625 x 0.625_301/"
FOLDER3 = "/run/media/neko/Archive_001/Archive/2024-09-02 DogCT/20240116/XH-001_20240116-135807-1914_140417/0.625 x 0.625_401/"
FOLDER4 = "/run/media/neko/Archive_001/Archive/2024-09-02 DogCT/20240117/XH-001_20240117-161233-1918_161310/0.625 x 0.625_201/"

for folder in [FOLDER1, FOLDER2, FOLDER3, FOLDER4]:
    dcm_interface.show_debug_all_file_in_folder(folder)