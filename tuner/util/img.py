# coding=utf-8

import matplotlib.pyplot as plt
import re
import seaborn

from tuner.util import file_helper


def draw(data_s, label_n):
    plt.plot(data_s)
    plt.plot([1, 5, 8])
    plt.ylabel(label_n)
    plt.show()


def mark():
    exp_file = file_helper.read2mem('../../line')
    exp_datas = exp_file.split('\n')
    cur_line = list()
    line_cnt = 0
    for exp_data in exp_datas:
        if exp_data.startswith('=='):
            plt.plot(cur_line)
            line_cnt += 1
            del cur_line[:]
        elif exp_data.startswith('['):
            # 超参数转图片名
            jpg_name = str(
                [num_str[0: 4].replace("-", "") for num_str in exp_data[1: -1].split(', ')]
            )[1: -1].replace("'", "").replace(', ', '-')
            plt.savefig(str(line_cnt / 2) + '-' + jpg_name + '.jpg')
            plt.clf()
            continue
        else:
            cur_line.append(exp_data)


def avg_line():
    exp_file = file_helper.read2mem('../../line')
    exp_datas = exp_file.split('\n')
    cur_sum = 0
    cur_cnt = 0
    avg_s = list()
    is_loss = True
    for exp_data in exp_datas:
        if exp_data.startswith('=='):
            if is_loss:
                avg_s.append(cur_sum / cur_cnt)
                cur_sum = 0
                cur_cnt = 0
            is_loss = not is_loss
            # 两次一翻转
        elif exp_data.startswith('['):
            continue
        elif re.match("^\d+?\.\d+?$", exp_data) is not None:
            if is_loss:
                cur_sum += float(exp_data)
                cur_cnt += 1
    plt.plot(avg_s)
    plt.show()

if __name__ == '__main__':
    mark()
    avg_line()