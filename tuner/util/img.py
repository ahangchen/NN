# coding=utf-8
import os

import matplotlib.pyplot as plt
import re
import seaborn
import shutil

from tuner.ctrl.const_define import HP_FILE_PATH, LINE_FILE_PATH, GRAD_FILE_PATH, PREDICT_FILE_PATH
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
            plt.savefig('data/' + str(line_cnt / 2) + '-' + jpg_name + '.png')
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
    plt.savefig('data/avg_loss_change.png')
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
    plt.savefig('data/hp_change.png')
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
    plt.savefig('data/grad_change.png')
    plt.show()


def viz_fit_hp():
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


def viz_fit_predict():
    actual_fit_str = file_helper.read2mem('../../' + LINE_FILE_PATH)
    actual_fit_s = actual_fit_str.split('===\n')
    actual_loss = list()
    predict_fit_str = file_helper.read2mem('../../' + PREDICT_FILE_PATH)
    predict_fit_s = predict_fit_str.split('===\n')
    predict_loss = list()
    cur_idx = 0
    print('actual_fit_s length: %d' % len(actual_fit_s))
    print('predict_fit_s length: %d' % len(predict_fit_s))
    for loss_data, predict_data in zip(actual_fit_s, predict_fit_s):
        loss_es = loss_data.split('\n')
        predict_s = predict_data.split('\n')
        print('loss_es length: %d' % len(loss_es))
        print('predict_s length: %d' % len(predict_s))
        for loss, predict in zip(loss_es, predict_s):
            if re.match("^\d+?\.\d+?$", loss) is not None:
                actual_loss.append(float(loss))
            if re.match("^\d+?\.\d+?$", predict) is not None:
                predict_loss.append(float(predict))
        if len(actual_loss) < 1 or len(predict_loss) < 1:
            print('actual_loss length: %d' % len(actual_loss))
            print('predict_loss length: %d' % len(predict_loss))
            continue
        actual_pl, = plt.plot(actual_loss, label='actual')
        predict_pl, = plt.plot(predict_loss, label='predict')
        plt.legend(handles=[actual_pl, predict_pl])
        plt.savefig('data/fit_result%d.png' % cur_idx)
        plt.clf()
        cur_idx += 1
        del actual_loss[:]
        del predict_loss[:]


def viz_hp2trend(dir_pos):
    fit_loss_es = file_helper.read2mem(dir_pos + 'hp2trend_fit_loss.txt').split('\n')[:-1]
    stable_loss_es_predict = file_helper.read2mem(dir_pos + 'hp2trend_stable_loss_predict.txt').split('\n')[:-1]
    stable_loss_es_label = file_helper.read2mem(dir_pos + 'hp2trend_stable_loss_label.txt').split('\n')[:-1]
    hps = list()
    for i in range(5):
        hps.append(file_helper.read2mem(dir_pos + 'hp2trend_hps%d.txt' % i).split('\n')[:-1])
    grads = list()
    for i in range(5):
        grads.append(file_helper.read2mem(dir_pos + 'hp2trend_grads%d.txt' % i).split('\n')[:-1])
    plt.plot(fit_loss_es, label='fit_loss')
    plt.savefig(dir_pos + 'hp2trend_fit_loss.png')
    plt.clf()
    del fit_loss_es[:]
    pl_stable_loss_predict, = plt.plot(stable_loss_es_predict, label='stable_loss_predict')
    pl_stable_loss_label, = plt.plot(stable_loss_es_label, label='stable_loss_label')
    plt.legend(handles=[pl_stable_loss_predict, pl_stable_loss_label])
    plt.savefig(dir_pos + 'stable_loss.png')
    plt.clf()
    del stable_loss_es_predict[:]
    del stable_loss_es_label[:]
    pl_list = list()
    for i in range(5):
        pl_hp, = plt.plot(hps[i], label='hp%d' % i)
        pl_list.append(pl_hp)
    plt.legend(handles=pl_list)
    plt.savefig(dir_pos + 'hps.png')
    plt.clf()
    del pl_list[:]
    for i in range(5):
        pl_grad, = plt.plot(grads[i], label='grad%d' % i)
        pl_list.append(pl_grad)
    plt.legend(handles=pl_list)
    plt.savefig(dir_pos + 'grads.png')
    plt.clf()
    del pl_list[:]


if __name__ == '__main__':
    viz_hp2trend('../../')