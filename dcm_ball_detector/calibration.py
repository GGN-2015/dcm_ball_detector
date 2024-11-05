import numpy as np
import itertools

# 输入一个 n * 3 的矩阵 marker_coord_matrix
# 每行表示一个标志物三维坐标，此函数会输出标志物之间两两的距离，作为一个向量
def get_marker_matrix_info(marker_coord_matrix: np.ndarray):
    assert isinstance(marker_coord_matrix, np.ndarray)
    assert len(marker_coord_matrix.shape) == 2
    assert marker_coord_matrix.shape[1] == 3 # 空间坐标，原点无所谓
    assert marker_coord_matrix.shape[0] >= 3 # 至少有三个标志物
    arr = []
    for i in range(marker_coord_matrix.shape[0]):
        for j in range(i + 1, marker_coord_matrix.shape[0]):
            vector1  = marker_coord_matrix[i]
            vector2  = marker_coord_matrix[j]
            distance = np.sqrt(np.sum((vector1 - vector2) ** 2))
            arr.append(distance)
    return np.array(arr)

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
def space_transfer(time_x_y_dict: dict, xdir_len_mm: float, dataset_meta_dict: dict) -> dict:
    time = time_x_y_dict["time"]
    xpos = time_x_y_dict["xpos"]
    ypos = time_x_y_dict["ypos"]
    slice_thickness = dataset_meta_dict["slice_thickness"]

    # 请注意，由于我们的图片进行过 xy 缩放，所以一定要注意要使用缩放后的比例尺才行
    pixel_spacing = np.array(dataset_meta_dict["resize_rate"]) * np.array(dataset_meta_dict["pixel_spacing"])
    return {
        "xmm": float(ypos * pixel_spacing[1]),
        "ymm": xdir_len_mm - float(xpos * pixel_spacing[0]),
        "zmm": float(time * slice_thickness),
    }

# 对一个 list 中的所有 time_x_y_dict 做 space_transfer
# 生成一个 list of dict
def space_transfer_all(time_x_y_dict_list: list, xdir_len_mm:float, dataset_meta_dict: dict) -> list:
    arr = []
    for time_x_y_dict in time_x_y_dict_list:
        arr.append(space_transfer(time_x_y_dict, xdir_len_mm, dataset_meta_dict))
    return arr

# 生成全排列
# 将来会用于根据距离比例，分析标志物顺序
def generate_permutations(n:int) -> list:
    elements = list(range(n))
    permutations = list(itertools.permutations(elements))
    return permutations

# 对二维矩阵进行按行置换
# 用于枚举标志物顺序的全排列
def perform_permutation(matrix_2d: np.ndarray, permutation_index_list: list) -> np.ndarray:
    arr = []
    for index in permutation_index_list:
        arr.append(list(matrix_2d[index]))
    return np.array(arr)

# 获取最优排布顺序
# 最后会还原回 zmm,xmm,ymm 的 list of dict 结构
def get_best_marker_order(standard_marker_coord_matrix: np.ndarray, ct_marker_coord_list: dict) -> list:
    ct_matrix = np.array([
        [
            ct_marker_coord["zmm"], # z 是扫描线前进的方向
            ct_marker_coord["xmm"], # x 和 y 是单张图像伸展方向
            ct_marker_coord["ymm"],
        ]
        for ct_marker_coord in ct_marker_coord_list
    ])
    std_matrix_info = get_marker_matrix_info(standard_marker_coord_matrix)
    score_message = []
    for permutation in generate_permutations(len(ct_marker_coord_list)):
        ct_matrix_now      = perform_permutation(ct_matrix, permutation)
        ct_matrix_info_now = get_marker_matrix_info(ct_matrix_now)
        score_message.append((
            np.sqrt(np.sum((std_matrix_info - ct_matrix_info_now) ** 2)), # 欧式距离越小越好
            ct_matrix_now
        ))
    score_message = sorted(score_message)
    best_marker_matrix = score_message[0][1]

    print(std_matrix_info) # 输出距离对比信息 DEBUG
    print(get_marker_matrix_info(best_marker_matrix))

    return [
        {
            "zmm": float(best_marker_matrix[i][0]),
            "xmm": float(best_marker_matrix[i][1]),
            "ymm": float(best_marker_matrix[i][2]),
        }
        for i in range(best_marker_matrix.shape[0])
    ]