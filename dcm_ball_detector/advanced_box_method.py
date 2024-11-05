# 基于 svm 的二次筛选过程
import functools
import json
import math
import numpy as np
from scipy.ndimage import label
from scipy.spatial import ConvexHull
from collections import Counter

from . import svm_utils
from . import image_log
from . import cube_get
from . import calibration
from . import dcm_interface
from . import stderr_log
from . import matplotlib_utils
from .os_interface import POS_IMAGE, NEG_IMAGE, INNER_POS_IMAGE, INNER_NEG_IMAGE

MAX_DIS_TOLERANCE   = 10   # 最大认同距离
MAX_TIME_TOLENRANCE = 4    # 最大认同时间间隔
MIN_LARGE_BALL_RATE = 0.15 # image3d 的 36 帧内容中，只要要有 20% 的大球，才会被认可为合法标志物
EMPTY_THRESH        = 5.5
TRY_CENTER_RADIUS   = 5
DEBUG_SHOW          = False # 输出凸包以及拟合图片

# 根据预先准备的数据集学习识别标志物的方法
# svm_1 用于区分一般图样和标志物图样
# svm_2 用于区分标志物图样出现时的大与小的不同
@functools.cache
def get_available_svm():
    svm_1 = svm_utils.get_svm_predictor(     POS_IMAGE,        NEG_IMAGE, False, "for ball detect")
    svm_2 = svm_utils.get_svm_predictor(INNER_POS_IMAGE, INNER_NEG_IMAGE, False, "for period detect")
    return svm_1, svm_2

# 使用二层 svm 对图像进行分类
# 区分三种不同状态：不是标志球，是标志球的小球阶段，是标志球的大球阶段
def svm_checker(log_numpy_array) -> str:
    svm1, svm2  = get_available_svm()
    image       = image_log.create_image_from_log_numpy_array(log_numpy_array)
    numpy_array = np.array(image)
    assert numpy_array.shape == log_numpy_array.shape
    numpy_array = numpy_array.flatten()
    if svm1.predict([numpy_array])[0] == 1: # 说明外层 svm 呈阴性
        tag = "is_not_ball"
    elif svm2.predict([numpy_array])[0] == 1: # 说明内层 svm 呈阴性
        tag = "is_small_ball"
    else:
        tag = "is_large_ball" # 说明内层外层都呈现阳性
    image_log.save_image_to_log_folder(image, subfolder=tag)
    return tag

# 对 36x36 的 log 对数数据进行预测
# 使用两个 svm 进行二分类，仅用于对大球的识别
def get_prediction_on_log_numpy_array(log_numpy_array):
    tag = svm_checker(log_numpy_array)
    return (tag == "is_large_ball")

# 寻找所有可能成为标志物的时刻以及矩形框
# 在先前筛选的基础上，进一步引入 SVM 进行筛选
def get_all_posible_box_in_folder(folder: str) -> list:
    dataset = cube_get.get_all_detected_picture_from_folder(folder)
    arr = []
    for item in dataset:
        timenow = item["timenow"] # 时间轴坐标
        box_rng = item["box_rng"] # 记录时间空间坐标
        image2d = item["image2d"]
        if get_prediction_on_log_numpy_array(image2d):
            arr.append({
                "timenow": timenow,
                "box_rng": box_rng
            })
    return arr

# 在有向图中加边
def __add_edge_in_dict(link: dict, i: int, j: int):
    assert i != j
    link[i].append(j)

# 给定一个 36x36 的矩形框
# 计算其中心位置, rnd 控制是否对位置进行取整
def get_center_pos_in_box_rng(box_rng, rnd=True):
    if rnd:
        return (round((box_rng[0] + box_rng[1])/2), round((box_rng[2] + box_rng[3])/2))
    else:
        return (((box_rng[0] + box_rng[1])/2), ((box_rng[2] + box_rng[3])/2))

# 计算两个点坐标的欧式距离
def get_distance(pos1, pos2) -> float:
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# 检查两个 36x36 的切片是否可以被归类为同一个球的不同出现时刻
def check_frame_in_same_object(frame1: dict, frame2: dict) -> bool:
    timenow_1 = frame1["timenow"]
    timenow_2 = frame2["timenow"]
    box_rng_1 = frame1["box_rng"]
    box_rng_2 = frame2["box_rng"]
    dis_check = (get_distance(get_center_pos_in_box_rng(box_rng_1), get_center_pos_in_box_rng(box_rng_2)) <= MAX_DIS_TOLERANCE)
    timecheck = (abs(timenow_1 - timenow_2) <= MAX_TIME_TOLENRANCE)
    return dis_check and timecheck

# 在某个节点开始 DFS 它所在的连通块
# 以 list 的形式返回它所在的连通块
def __dfs(x: int, vis: dict, link: dict):
    assert vis[x] is False
    arr = [x]
    vis[x] = True
    for y in link[x]: # 枚举所有从 x 出发能走到的节点
        if not vis[y]:
            arr += __dfs(y, vis, link)
    return arr

# 在一个无向图中寻找所有连同快
# 并最终返回所有连通块构成的集合
def get_cluster_set_in_graph(n: int, link: dict):
    vis = {}
    arr = []
    for i in range(n): # 初始化 vis 数组
        vis[i] = False
    for i in range(n):
        if not vis[i]:
            arr.append(__dfs(i, vis, link))
    vcnt = sum([len(cluster) for cluster in arr])
    assert vcnt == n # 确保每个节点都被恰好访问过一次
    return arr

# 对所有筛选出的举行框框进行聚类
# 把时空距离较近的矩形框合并为一个等价类
# 未来还要再寻找到等价类中最靠中间、最可能成为实际中心点的元素，但是这个函数不负责做这个事情
def get_all_cluster_in_folder(folder: str) -> list:
    link = {}
    object_list = get_all_posible_box_in_folder(folder) # 先获得所有可能成为答案的时刻切片
    for i in range(len(object_list)): # 初始化邻接表
        link[i] = []
    for i in range(len(object_list)):
        for j in range(i):
            if check_frame_in_same_object(object_list[i], object_list[j]): # 绘制有向图
                __add_edge_in_dict(link, i, j)
                __add_edge_in_dict(link, j, i)
    # 通过对图进行 DFS 确定所有聚类
    cluster_set_idx = get_cluster_set_in_graph(len(object_list), link)
    cluster_set_frm = [
        [object_list[idx] for idx in cluster_idx] for cluster_idx in cluster_set_idx
    ] # 根据编号聚类构建 frame 聚类
    return cluster_set_frm

# 计算中位数
def calculate_median(numbers: list):
    numbers.sort()               # 先对列表进行排序
    n = len(numbers)    
    if n % 2 == 0:               # 如果列表长度为偶数，中位数为中间两个数的平均值
        median = (numbers[n // 2 - 1] + numbers[n // 2]) / 2
    else:                        # 如果列表长度为奇数，中位数为中间的数
        median = numbers[n // 2]
    return median

# 获取某个指定聚类的聚类时空中心
# 但是只是粗略的获取，可能需要进一步的算法上的修正（尽管是粗略的获取我个人感觉已经能做到 2mm 以内的精确程度了）
def get_cluster_set_center(cluser_frm: list) -> dict:
    assert len(cluser_frm) > 0 # 不能求空 cluster 的中心点
    t_lis = []
    x_lis = []
    y_lis = []
    for frame in cluser_frm:
        timenow = frame["timenow"]
        box_rng = frame["box_rng"]
        x_avg, y_avg = get_center_pos_in_box_rng(box_rng)
        x_lis.append(x_avg)
        y_lis.append(y_avg)
        t_lis.append(timenow)
    return {
        "time": round(calculate_median(t_lis)), # 这里需要取整的主要原因是，中位数可能会有 .5 的情况
        "xpos": round(calculate_median(x_lis)),
        "ypos": round(calculate_median(y_lis))
    }

# 给定一个聚类的中心点
# 返回这个聚类中心点对应的时空邻域中，有多少比例的时间片段被认为是大球
@functools.cache
def get_cluster_center_large_ball_rate(folder: str, time: int, xpos: int, ypos: int):
    image3d = cube_get.get_cube_from_log_numpy_list_in_folder_around_center(folder, time, xpos, ypos)
    tlen, xlen, ylen = image3d.shape
    large_ball_cnt = 0
    for i in range(tlen):
        if svm_checker(image3d[i]) == "is_large_ball":
            large_ball_cnt += 1
    return round(large_ball_cnt/tlen, ndigits=4)

# 计算众数，如果有多个，任取一个
def find_mode(data):
    count = Counter(data)
    mode = count.most_common(1)[0][0]
    return mode

# 计算中心点所在连通块的编号
# 修正中心点可能为空白的情况
def get_center_label_in_labled_array(center, labeled_array):
    arr = []
    for i in range(-TRY_CENTER_RADIUS, TRY_CENTER_RADIUS):
        for j in range(-TRY_CENTER_RADIUS, TRY_CENTER_RADIUS):
            arr.append(labeled_array[center[0] + i, center[1] + j])
    return find_mode(arr)

# 构建 AABB 包围盒
def get_xyrange_from_hull(hull_array2d):
    assert hull_array2d.shape[1] == 2
    xmin, xmax, ymin, ymax = hull_array2d[0][0], hull_array2d[0][0], hull_array2d[0][1], hull_array2d[0][1]
    for i in range(hull_array2d.shape[0]):
        xmin = min(xmin, hull_array2d[i][0])
        xmax = max(xmax, hull_array2d[i][0])
        ymin = min(ymin, hull_array2d[i][1])
        ymax = max(ymax, hull_array2d[i][1])
    return xmin, xmax, ymin, ymax

# 给定多边形的顶点信息，计算多边形的重心
def compute_centroid(points):
    x = points[:, 1]  # x 坐标
    y = points[:, 0]  # y 坐标
    n = len(points)
    A   = 0.0  # 面积
    C_x = 0.0  # 重心 x
    C_y = 0.0  # 重心 y
    for i in range(n):
        j = (i + 1) % n  # 处理闭合多边形
        factor = x[i] * y[j] - x[j] * y[i]
        A += factor
        C_x += (x[i] + x[j]) * factor
        C_y += (y[i] + y[j]) * factor
    A *= 0.5
    C_x /= (6 * A)
    C_y /= (6 * A)
    return C_x, C_y

# 返回修正性中心的偏移坐标，以像素单位长度为单位
# 返回 0, 0 表示不偏移
# 还没实现
def get_fixed_xypos_for_image2d(image2d: np.ndarray):
    labeled_array, num_features = label(image2d)
    center              = (labeled_array.shape[0] // 2, labeled_array.shape[1] // 2)
    center_label        = get_center_label_in_labled_array(center, labeled_array)
    connected_component = np.argwhere(labeled_array == center_label)
    hull        = ConvexHull(connected_component)    # 计算凸包
    hull_points = connected_component[hull.vertices] # 获得凸包每个点的二维坐标
    centroid    = compute_centroid(hull_points)      # 计算多边形重心
    xmin, xmax, ymin, ymax = get_xyrange_from_hull(hull_points)
    if (xmin < TRY_CENTER_RADIUS or 
        ymin < TRY_CENTER_RADIUS or 
        xmax >= image2d.shape[0] - TRY_CENTER_RADIUS or 
        ymax >= image2d.shape[1] - TRY_CENTER_RADIUS): # 发现了连通块过大的情况
        return None, None
    if DEBUG_SHOW:
        matplotlib_utils.debug_output_for_connected_component(connected_component, hull, centroid)
    xpos, ypos = centroid - np.array(center)
    return np.array([[xpos], [ypos]]), hull.area # 返回新中心到旧中心的相对坐标

# 修正 z 坐标信息
def get_fixed_zpos_for_areas(areas):
    x_data = []
    y_data = []
    for z in range(len(areas)): # 提取坐标点信息
        if areas[z] is not None:
            x_data.append(z)
            y_data.append(areas[z])
    coefficients = np.polyfit(x_data, y_data, 2)  # 拟合一个二次函数
    a, b, c      = coefficients # 提取拟合的系数 a, b, c
    pivot        = - b / (2*a)    # 计算二次函数对称轴
    if DEBUG_SHOW:
        matplotlib_utils.debug_output_curve(x_data, y_data, a, b, c)
    return pivot - cube_get.T_RADIUS

# 修正图像中计算得到的球中心点的 xy 坐标
def get_fixed_xyzpos_for_center(folder: str, time: int, xpos: int, ypos: int):
    image3d = cube_get.get_cube_from_log_numpy_list_in_folder_around_center(folder, time, xpos, ypos)
    tlen, xlen, ylen = image3d.shape
    centers = [] # 所有修正性中心放到一个 list 里
    areas   = [] # 计算所有类似圆的结构的面积
    for i in range(tlen):
        if svm_checker(image3d[i]) in ["is_large_ball", "is_small_ball"]: # 对于所有大球，计算修正性中心
            image_now = image3d[i].copy()
            image_now[image_now > EMPTY_THRESH] = 1    # 二值化
            optional_pos, area = get_fixed_xypos_for_image2d(image_now) # 计算 xy 坐标的修正位置
            if optional_pos is not None:
                centers.append(optional_pos)
            areas.append(area)
        else:
            areas.append(None) # 未知面积信息
    dz = get_fixed_zpos_for_areas(areas)
    if len(centers) > 0:
        stacked_array = np.stack(centers)
        mean_array = np.mean(stacked_array, axis=0)
        return mean_array[0][0], mean_array[1][0], dz # 返回坐标偏移量
    else:
        return 0, 0, dz # 如果没有合法大圆阶段，跳过，但是这种情况几乎不会出现

# 给定一个已知可能成为聚类中心的点
# 获取它的时空邻域 image3d 对象，检查大球的存在率
# 如果小于一个指定的阈值，则认为是假阳
def check_cluster_center_large_ball_rate_ok(folder: str, cluster_set_center):
    time = cluster_set_center["time"]
    xpos = cluster_set_center["xpos"]
    ypos = cluster_set_center["ypos"]
    return get_cluster_center_large_ball_rate(folder, time, xpos, ypos) >= MIN_LARGE_BALL_RATE

# 获得所有聚类的中心点
# 后期可能还需要进行进一步的坐标矫正，但是将来再说
# show_rate=True 用于展示时空邻域中的大圆比例
def get_all_cluster_center_in_folder(folder: str, show_rate=False) -> list:
    cluster_set_frm = get_all_cluster_in_folder(folder)
    cluster_center_set = []
    for cluser_frm in cluster_set_frm: # 计算出每个聚类的中心点
        cluster_set_center = get_cluster_set_center(cluser_frm)
        if check_cluster_center_large_ball_rate_ok(folder, cluster_set_center): # 如果大圆率符合要求
            time    = cluster_set_center["time"]
            xpos    = cluster_set_center["xpos"]
            ypos    = cluster_set_center["ypos"]
            new_obj = json.loads(json.dumps(cluster_set_center))
            if show_rate: # 是否记录大圆比例
                new_obj["rate"] = get_cluster_center_large_ball_rate(folder, time, xpos, ypos)
            cluster_center_set.append(new_obj)
    for cluster_center in cluster_center_set: # 计算 xy 坐标修正
        time       = cluster_center["time"]
        xpos       = cluster_center["xpos"]
        ypos       = cluster_center["ypos"]
        dx, dy, dz = get_fixed_xyzpos_for_center(folder, time, xpos, ypos) # 中心点 xyz 坐标修正
        cluster_center["xpos"] += dx
        cluster_center["ypos"] += dy
        cluster_center["time"] += dz
    return cluster_center_set

# 计算出以毫米为单位的空间坐标信息
# 然后再根据标志物配置信息调节标志物的相对顺序
# !!! 未测试 !!!
def get_all_marker_center_and_give_the_mm_coord_with_correct_order(marker_coord_matrix: np.ndarray, folder: str):
    cluster_center_set   = get_all_cluster_center_in_folder(folder)
    if len(cluster_center_set) != len(marker_coord_matrix):
        stderr_log.log_error("mutiple marker sets or incomplete marker set are currently not supported.")
        exit(1)
    ct_marker_coord_list = calibration.space_transfer_all(
        cluster_center_set,
        dcm_interface.get_metadata_from_dcm_folder(folder)
    )
    calibration.get_best_marker_order(marker_coord_matrix, ct_marker_coord_list)
    return cluster_center_set