import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage import io

try:
    from . import stderr_log
except:
    import stderr_log

# 指定两个装满图片的文件夹
# 使用 rbf 核函数的支持向量机训练一个二分类器
def get_svm_predictor(pos_folder, neg_folder, need_report=False):
    stderr_log.log_info("preparing svm.")
    images = [] # 加载图像和标签
    labels = []
    for filename in os.listdir(pos_folder): # 加载类别1图像
        if filename.endswith('.png'):
            img_path = os.path.join(pos_folder, filename)
            img = io.imread(img_path)
            img = img.flatten()  # 将36x36图像展平为一维数组
            images.append(img)
            labels.append(0)  # 类别1的标签为0
    for filename in os.listdir(neg_folder): # 加载类别2图像
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
    y_pred = classifier.predict(X_test) # 进行预测
    if need_report:
        print(classification_report(y_test, y_pred)) # 输出分类报告
    return classifier

if __name__ == "__main__":
    dirnow = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirnow)
    get_svm_predictor("./ground_truth/pos", "./ground_truth/neg", True)
    get_svm_predictor("./ground_truth/inner_pos/pos", "./ground_truth/inner_pos/neg", True)