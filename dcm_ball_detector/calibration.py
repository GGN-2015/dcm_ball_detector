import numpy as np

# 输入一个 n * 3 的矩阵
# 每行表示一个标志物三维坐标，此函数会输出标志物之间两两的距离，作为一个向量
# 然后，距离以第一个球和第二个球之间的距离为单位距离
def get_marker_matrix_info(marker_list):
    assert isinstance(marker_list, np.ndarray)
    assert len(marker_list.shape) == 2
    assert marker_list.shape[1] == 3 # 空间坐标，原点无所谓
    assert marker_list.shape[0] >= 3 # 至少有三个标志物
    arr = []
    for i in range(marker_list.shape[0]):
        for j in range(i + 1, marker_list.shape[0]):
            vector1 = marker_list[i]
            vector2 = marker_list[j]
            arr.append(np.sqrt(np.sum((vector1 - vector2) ** 2)))
    return np.array(arr) / arr[0]

def test_get_marker_matrix_info(): # 对距离计算函数进行测试
    sample_matrix = np.array([
        [ 0.0000000000000000,  0.0000000000000000, 0.0000000000000000], 
        [69.3940693200773921, -0.0000000000000036, 0.0000000000000000], 
        [ 9.1217166176732363, 56.4598693992500529, 0.0000000000000001], 
        [72.4396220656959713, 50.7199986380889953, 0.3294267527434576], 
    ])
    print(get_marker_matrix_info(sample_matrix))

# 借助来自于数据集的元信息中的 slice_thickness 以及 pixel_spacing
# 根据 CT 图像坐标体元编号，变换为以毫米为单位空间直角坐标系下（坐标原点不变）
def space_transfer(time_x_y_dict: dict, dataset_meta_dict: dict) -> dict:
    time = time_x_y_dict["time"]
    xpos = time_x_y_dict["xpos"]
    ypos = time_x_y_dict["ypos"]
    slice_thickness = dataset_meta_dict["dataset_meta_dict"]
    pixel_spacing   = dataset_meta_dict["pixel_spacing"]
    return {
        "zmm": time * slice_thickness,
        "xmm": xpos * pixel_spacing,
        "ymm": ypos * pixel_spacing,
    }