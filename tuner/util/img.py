# coding=utf-8
import os

import matplotlib.pyplot as plt
import re
import seaborn
import shutil

from tuner.ctrl.const_define import HP_FILE_PATH, LINE_FILE_PATH, GRAD_FILE_PATH
from tuner.util import file_helper


def draw(data_s, label_n):
    plt.plot(data_s)
    plt.plot([1, 5, 8])
    plt.ylabel(label_n)
    plt.show()


def mark():
    exp_file = file_helper.read2mem('../../' + LINE_FILE_PATH)
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
                [num_str[0: 4].replace("-", "") for num_str in exp_data[1: -1].split()]
            )[1: -1].replace("'", "").replace(', ', '-')
            plt.savefig('data/' + str(line_cnt / 2) + '-' + jpg_name + '.jpg')
            plt.clf()
            continue
        else:
            cur_line.append(exp_data)


def avg_line():
    exp_file = file_helper.read2mem('../../' + LINE_FILE_PATH)
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
    plt.savefig('data/avg_loss_change.jpg')
    plt.show()


def hp_line():
    batch_size_s = list()
    depth_s = list()
    num_hidden_s = list()
    layer_cnt_s = list()
    patch_size_s = list()
    exp_file = file_helper.read2mem('../../' + HP_FILE_PATH)
    # print(exp_file)
    exp_datas = exp_file.split('\n')
    for exp_data in exp_datas:
        hps = exp_data[1: -1].split(', ')
        if re.match("^\d+?\.\d+?$", hps[0]) is None:
            continue
        batch_size_s.append(float(hps[0]))
        depth_s.append(float(hps[1]))
        num_hidden_s.append(float(hps[2]))
        layer_cnt_s.append(float(hps[3]))
        patch_size_s.append(float(hps[4]))
    batch_pl, = plt.plot(batch_size_s, label='batch_size')
    depth_pl, = plt.plot(depth_s, label='depth')
    num_pl, = plt.plot(num_hidden_s, label='num_hidden')
    layer_pl, = plt.plot(layer_cnt_s, label='layer_cnt')
    patch_pl, = plt.plot(patch_size_s, label='patch_size')
    plt.legend(handles=[batch_pl, depth_pl, num_pl, layer_pl, patch_pl])
    plt.savefig('data/hp_change.jpg')
    plt.show()


def grad_line():
    batch_size_s = list()
    depth_s = list()
    num_hidden_s = list()
    layer_cnt_s = list()
    patch_size_s = list()
    exp_file = file_helper.read2mem('../../' + GRAD_FILE_PATH)
    # print(exp_file)
    exp_datas = exp_file.split('\n')
    for exp_data in exp_datas:
        hps = exp_data[1: -1].split(', ')
        if re.match("^(-)\d+?\.\d+?$", hps[0]) is None:
            continue
        batch_size_s.append(float(hps[0]))
        depth_s.append(float(hps[1]))
        num_hidden_s.append(float(hps[2]))
        layer_cnt_s.append(float(hps[3]))
        patch_size_s.append(float(hps[4]))
    batch_pl, = plt.plot(batch_size_s, label='batch_size')
    depth_pl, = plt.plot(depth_s, label='depth')
    num_pl, = plt.plot(num_hidden_s, label='num_hidden')
    layer_pl, = plt.plot(layer_cnt_s, label='layer_cnt')
    patch_pl, = plt.plot(patch_size_s, label='patch_size')
    plt.legend(handles=[batch_pl, depth_pl, num_pl, layer_pl, patch_pl])
    plt.savefig('data/grad_change.jpg')
    plt.show()

if __name__ == '__main__':
    mark()
    avg_line()
    hp_line()
    grad_line()
    shutil.move('../../' + LINE_FILE_PATH, 'data/' + LINE_FILE_PATH)
    shutil.move('../../' + HP_FILE_PATH, 'data/' + HP_FILE_PATH)
    shutil.move('../../' + GRAD_FILE_PATH, 'data/' + GRAD_FILE_PATH)
    shutil.move('../../checkpoint', 'data/checkpoint')
    shutil.move('../../model.ckpt', 'data/model.ckpt')
    shutil.move('../../model.ckpt.meta', 'data/model.ckpt.meta')
