import sys
import re

COLOR_DICT = { # 记录颜色编号与消息类型的对应关系
    "info": 34,
    "error": 31,
    "warning": 33,
    "tips": 35,
}

# 根据消息类型获取颜色编号
def get_color_by_msg_type(msg_type):
    assert COLOR_DICT.get(msg_type) is not None
    return COLOR_DICT[msg_type]

# 用于在信息中提供颜色显示
def color_info_embedding(msg: str) -> str:
    pattern = r"<<<(\d+)\[([^\]]*)\]>>>"
    new_msg = msg
    for item in re.finditer(pattern, msg): # 颜色带入
        new_msg = new_msg.replace(item[0], "\033[1;%sm%s\033[0m" % (item[1], item[2]))
    return new_msg

# 对信息中需要突出强调的部分进行预处理
def preprocess_msg(msg: str) -> str:
    msg = color_info_embedding(msg)
    return msg

# 向标准错误流中写入数据
def general_log(msg_type, msg):
    color_id = get_color_by_msg_type(msg_type)
    full_msg = "\033[1;%dm%7s\033[0m: %s\n" % (color_id, msg_type, preprocess_msg(msg.rstrip()))
    sys.stderr.write(full_msg)

# 向标准错误流写入消息
def log_info(msg):
    general_log("info", msg)

# 向标准错误流写入错误
def log_error(msg):
    general_log("error", msg)

# 向标准错误流写入警告
def log_warning(msg):
    general_log("warning", msg)

# 向标准错误流写入提示信息
def log_tips(msg):
    general_log("tips", msg)