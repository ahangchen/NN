# coding=utf-8

import matplotlib.pyplot as plt
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


def loss_line():
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

if __name__ == '__main__':
    mark()