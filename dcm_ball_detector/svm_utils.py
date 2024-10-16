import os
import numpy as np
import functools
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from skimage import io

try:
    from . import stderr_log
except:
    import stderr_log

SVM_ACC_THRESH = 0.96 # SVM 的准确率应该至少达到 96%

# 递归获取文件夹中的所有文件
# 用于读取 ground truth 数据
@functools.cache
def list_all_file_recursively_in_folder(folder: str) -> list:
    arr = []
    for file in os.listdir(folder):
        file_now = os.path.join(folder, file)
        if os.path.isdir(file_now):
            arr += list_all_file_recursively_in_folder(file_now)
        else:
            arr.append(file_now)
    return arr

# 指定两个装满图片的文件夹
# 使用 rbf 核函数的支持向量机训练一个二分类器
@functools.cache
def get_svm_predictor(pos_folder, neg_folder, need_report=False, name=""):
    if name == "":
        stderr_log.log_info("preparing svm.")
    else:
        stderr_log.log_info("preparing svm %s." % name)
    images = [] # 加载图像和标签
    labels = []
    for filename in list_all_file_recursively_in_folder(pos_folder): # 加载类别1图像
        if filename.endswith('.png'):
            img_path = os.path.join(pos_folder, filename)
            img = io.imread(img_path)
            img = img.flatten()  # 将36x36图像展平为一维数组
            images.append(img)
            labels.append(0)  # 类别1的标签为0
    for filename in list_all_file_recursively_in_folder(neg_folder): # 加载类别2图像
        if filename.endswith('.png'):
            img_path = os.path.join(neg_folder, filename)
            img = io.imread(img_path)
            img = img.flatten()  # 将36x36图像展平为一维数组
            images.append(img)
            labels.append(1)  # 类别2的标签为1
    X = np.array(images) # 转换为numpy数组
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 划分训练集和测试集
    classifier = svm.SVC(kernel='rbf', random_state=42) # 创建并训练支持向量机分类器
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  # 进行预测
    acc = accuracy_score(y_test, y_pred) # 计算准确率
    stderr_log.log_info("getting svm with acc = <<<32[%5.2f%%]>>>." % (100 * acc)) # green output
    assert acc >= SVM_ACC_THRESH # 低于 svm 准确率阈值直接报错，不要继续运行算法
    if need_report:
        print(classification_report(y_test, y_pred)) # 输出分类报告
    return classifier

if __name__ == "__main__":
    dirnow = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirnow)
    get_svm_predictor("./ground_truth/pos", "./ground_truth/neg", True)
    get_svm_predictor("./ground_truth/inner_pos/pos", "./ground_truth/inner_pos/neg", True)