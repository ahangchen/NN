# coding=utf-8
import random

import numpy as np
import tensorflow as tf
import os
import math


def assign_diffable_vars2tensor(placeholder, input_size):
    # 为了赋值
    placeholder_piece_s = tf.split(0, input_size, placeholder)
    empty_var_inputs = [tf.Variable(0.0, name='init_hp_%d' % i, trainable=False) for i in range(input_size)]
    variable_info_input = [var_init_hyper.assign(tf.reshape(placeholder_piece_s[i], shape=()))
                           for i, var_init_hyper in enumerate(empty_var_inputs)]
    # 为了求导
    diffable_vars = [tf.Variable(init_hp, name='hp_%d' % i) for i, init_hp in enumerate(variable_info_input)]
    # 为了作为整个tensor运算
    tf_info_var_1d = tf.pack(diffable_vars)
    # 为了作为矩阵
    tf_info_var_2d = tf.reshape(tf_info_var_1d, [input_size, 1])
    return tf_info_var_2d


def tensor_reshape_with_matrix(input_matrix, input_shape, output_shape):
    # 一般从二维到一维
    output_weight = tf.Variable(tf.truncated_normal([input_shape[1], output_shape[0]], mean=-0.1, stddev=0.1))
    output_biase = tf.Variable(tf.truncated_normal([output_shape[0]]))
    output = tf.nn.xw_plus_b(input_matrix, output_weight, output_biase)
    return output


def dnn(input_tensor, input_shape, output_shape, drop_out=False, layer_cnt=2):
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
    y0 = tf.matmul(input_tensor, weights1) + biases1
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


def grad_optimizer(var_list, loss):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        0.5, global_step, 50, 0.8, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, return_v = zip(*optimizer.compute_gradients(loss, var_list=var_list))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, return_v), global_step=global_step)
    return optimizer, gradients


class FitTrendModel(object):
    def __init__(self, input_size, output_size):
        rnn_step_cnt = 10
        self.graph = tf.Graph()
        self.hyper_cnt = input_size
        with self.graph.as_default():
            # 接收输入
            self.ph_hypers = tf.placeholder(tf.float32, shape=[self.hyper_cnt], name='ph_hypers')
            self.tf_hypers = assign_diffable_vars2tensor(self.ph_hypers, self.hyper_cnt)
            # 先喂给一个神经网络产生input
            trend_input = dnn(self.tf_hypers, [self.hyper_cnt, 1], [output_size])
            # 通过一个RNN
            trend_outputs = rnn(tf.concat(0, trend_input), rnn_step_cnt, output_size)
            # RNN接一个小型NN
            trend_output = tensor_reshape_with_matrix(tf.concat(0, trend_outputs), [output_size, rnn_step_cnt],
                                                      [output_size])
            # 实际的trend
            self.train_label = [tf.placeholder(tf.float32, shape=[output_size])]
            # 预测准确率
            predict_accuracy = tf.reduce_mean(tf.sqrt(tf.square(tf.sub(trend_output, tf.concat(0, self.train_label)))))
            # 稳定时损失
            stable_loss = trend_outputs[-1]
            self.is_fit = tf.placeholder(tf.bool)
            self.loss = tf.cond(self.is_fit, lambda: predict_accuracy, lambda: stable_loss)

            # 优化器
            var_s = tf.trainable_variables()
            self.v_hp_s = var_s[0: self.hyper_cnt - 1]
            self.v_fit_s = [v for v in var_s if v not in self.v_hp_s]

            optimizer_fit, _ = grad_optimizer(self.v_fit_s, self.loss)
            optimizer_hp, self.grad_hps = grad_optimizer(self.v_hp_s, self.loss)

            self.optimizer = tf.cond(self.is_fit, lambda: optimizer_fit, lambda: optimizer_hp)

            self.saver = tf.train.Saver()
        self.session = tf.Session(graph=self.graph)

    def init(self):
        with self.session:
            tf.initialize_all_variables().run()

    def fit(self, input_data, trend):
        fit_dict = dict()
        fit_dict[self.is_fit] = True
        fit_dict[self.ph_hypers] = input_data
        fit_dict[self.train_label] = trend

        with self.session as fit_session:
            _, loss, hp_s = fit_session.run([self.optimizer, self.loss, self.grad_hps], feed_dict=fit_dict)
        print(loss)
        print(hp_s)


def init_model(input_size, output_size):
    model = FitTrendModel(input_size, output_size)
    model.init()
    return model


def fit_trend(model, input_data, trend):
    model.fit(input_data, trend)


def train_hp():
    pass
