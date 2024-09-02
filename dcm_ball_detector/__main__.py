from . import dcm_interface

FOLDER1 = "/run/media/neko/Archive_001/Archive/2024-09-02 DogCT/20240103/XH-1-0103_20240103-145148-1892_145219/0.625 x 0.625_301/"
FOLDER2 = "/run/media/neko/Archive_001/Archive/2024-09-02 DogCT/20240103/XH-2_20240103-152620-1893_152720/0.625 x 0.625_201"
FOLDER3 = "/run/media/neko/Archive_001/Archive/2024-09-02 DogCT/20240103/XH-2_20240103-152620-1893_152720/0.625 x 0.625_401"
FOLDER4 = "/run/media/neko/Archive_001/Archive/2024-09-02 DogCT/20240103/XH-2_20240103-152620-1893_152720/0.625 x 0.625_501"
THRESH = 0.5

for folder in [FOLDER1, FOLDER2, FOLDER3, FOLDER4]:
    dcm_interface.show_range_matrix_for_folder(folder, THRESH)