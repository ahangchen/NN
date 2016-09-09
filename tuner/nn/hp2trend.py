# coding=utf-8
import random

import numpy as np
import tensorflow as tf
import os
import math


def dnn(input, input_shape, output_shape, drop_out=False, layer_cnt=2):
    # input shape 是一个二维数组
    hidden_node_count = 1024
    # start weight
    hidden_stddev = np.sqrt(2.0 / input_shape[1])
    weights1 = tf.Variable(
        tf.truncated_normal([input_shape[1], hidden_node_count], stddev=hidden_stddev))
    biases1 = tf.Variable(tf.zeros([hidden_node_count]))
    # middle weight
    weights = []
    biases = []
    hidden_cur_cnt = hidden_node_count
    for i in range(layer_cnt - 2):
        if hidden_cur_cnt > 2:
            hidden_next_cnt = int(hidden_cur_cnt / 2)
        else:
            hidden_next_cnt = 2
        hidden_stddev = np.sqrt(2.0 / hidden_cur_cnt)
        weights.append(
            tf.Variable(tf.truncated_normal([hidden_cur_cnt, hidden_next_cnt], stddev=hidden_stddev)))
        biases.append(tf.Variable(tf.zeros([hidden_next_cnt])))
        hidden_cur_cnt = hidden_next_cnt
    # first wx + b
    y0 = tf.matmul(input, weights1) + biases1
    # first relu
    hidden = tf.nn.relu(y0)
    hidden_drop = hidden
    # first DropOut
    keep_prob = 0.5
    if drop_out:
        hidden_drop = tf.nn.dropout(hidden, keep_prob)

    # middle layer
    for i in range(layer_cnt - 2):
        y1 = tf.matmul(hidden_drop, weights[i]) + biases[i]
        hidden_drop = tf.nn.relu(y1)
        if drop_out:
            keep_prob += 0.5 * i / (layer_cnt + 1)
            hidden_drop = tf.nn.dropout(hidden_drop, keep_prob)

        y0 = tf.matmul(hidden, weights[i]) + biases[i]
        hidden = tf.nn.relu(y0)

    # last weight
    weights2 = tf.Variable(tf.truncated_normal([hidden_cur_cnt, output_shape[0]], stddev=hidden_stddev / 2))
    biases2 = tf.Variable(tf.zeros([output_shape[0]]))
    # last wx + b
    output = tf.matmul(hidden, weights2) + biases2
    return output


def rnn(x, n_steps, n_input, n_hidden=128):
    x = tf.reshape(x, [-1, n_steps, n_input])  # (batch_size, n_steps, n_input)
    # # permute n_steps and batch_size
    x = tf.transpose(x, [1, 0, 2])
    # # Reshape to prepare input to hidden activation
    x = tf.reshape(x, [-1, n_input])  # (n_steps*batch_size, n_input)
    # # Split data because rnn cell needs a list of inputs for the RNN inner loop
    x = tf.split(0, n_steps, x)  # n_steps * (batch_size, n_input)

    # Define a GRU cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    # Get lstm cell output
    y, final_state = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
    return y


def init_model(input_size, output_size):
    rnn_step_cnt = 10
    graph = tf.Graph()
    hyper_cnt = input_size
    with graph.as_default():
        # Input, Forget, Memory, Output gate: input, previous output, and bias.
        is_fit = tf.placeholder(tf.bool)
        # 接收输入
        ph_hypers = tf.placeholder(tf.float32, shape=[hyper_cnt], name='ph_hypers')

        # 为了赋值
        cnn_raw_hypers = tf.split(0, hyper_cnt, ph_hypers)
        var_init_hypers = [tf.Variable(0.0, name='init_hp_%d' % i, trainable=False) for i in range(hyper_cnt)]
        tf_init_hypers = [var_init_hyper.assign(tf.reshape(cnn_raw_hypers[i], shape=()))
                          for i, var_init_hyper in enumerate(var_init_hypers)]
        # 为了求导
        var_hypers = [tf.Variable(init_hp, name='hp_%d' % i) for i, init_hp in enumerate(tf_init_hypers)]
        # 为了作为整个tensor运算
        pack_var_hypers = tf.pack(var_hypers)
        tf_hypers = tf.reshape(pack_var_hypers, [hyper_cnt, 1])
        # 先喂给一个神经网络产生input
        trend_input = dnn(tf_hypers, [hyper_cnt, 1], [output_size])
        trend_output = rnn(trend_input, rnn_step_cnt, output_size)
        train_label = [tf.placeholder(tf.float32, shape=[output_size],
                                       name='train_labels{i}'.format(i=i)) for i in range(batch_cnt_per_step)]

    pass


def fit_trend():
    pass


def train_hp():
    pass
