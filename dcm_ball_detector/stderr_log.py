import sys

COLOR_DICT = { # 记录颜色编号与消息类型的对应关系
    "info": 34,
    "error": 31,
    "warning": 33,
}

# 根据消息类型获取颜色编号
def get_color_by_msg_type(msg_type):
    assert COLOR_DICT.get(msg_type) is not None
    return COLOR_DICT[msg_type]

# 向标准错误流中写入数据
def general_log(msg_type, msg):
    color_id = get_color_by_msg_type(msg_type)
    full_msg = "\033[1;%dm%7s\033[0m: %s\n" % (color_id, msg_type, msg.rstrip())
    sys.stderr.write(full_msg)

# 向标准错误流写入消息
def log_info(msg):
    general_log("info", msg)

# 向标准错误流写入错误
def log_error(msg):
    general_log("error", msg)

# 向标准错误流写入警告
def log_error(msg):
    general_log("warning", msg)