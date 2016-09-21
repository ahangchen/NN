# coding=utf-8
import os
import numpy as np
import tensorflow as tf


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
    #
    reset_hps = []
    for var in empty_var_inputs:
        reset_hps.append(var)
    for var in diffable_vars:
        reset_hps.append(var)
    return tf_info_var_2d, reset_hps


def tensor_reshape_with_matrix(input_matrix, output_shape):
    # 一般从二维到一维
    input_shape = input_matrix.get_shape()
    left_weight = tf.Variable(tf.truncated_normal([output_shape[0], int(input_shape[0])]))
    left_biase = tf.Variable(tf.truncated_normal([int(input_shape[1])]))
    left_output = tf.nn.xw_plus_b(left_weight, input_matrix, left_biase)
    left_output = tf.nn.relu(left_output)
    right_weight = tf.Variable(tf.truncated_normal([int(input_shape[1]), output_shape[1]]))
    right_biase = tf.Variable(tf.truncated_normal([output_shape[1]]))
    right_output = tf.nn.xw_plus_b(left_output, right_weight, right_biase)
    return right_output


def dnn(input_tensor, output_shape, drop_out=False, layer_cnt=2):
    input_shape = input_tensor.get_shape()
    input_shape = [int(shape_i) for shape_i in input_shape]
    # input shape 是一个二维数组
    hidden_node_count = 64
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
    # first relu6
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
    output = tensor_reshape_with_matrix(hidden, [output_shape[0], output_shape[1]])
    # last weight
    # weights2 = tf.Variable(tf.truncated_normal([hidden_cur_cnt, output_shape[0]], stddev=hidden_stddev / 2))
    # biases2 = tf.Variable(tf.zeros([output_shape[0]]))
    # last wx + b
    # output = tf.matmul(hidden, weights2) + biases2
    return output


def rnn(x, n_hidden=64):
    n_input = int(x.get_shape()[0])
    n_steps = int(x.get_shape()[1])
    x = tf.reshape(x, [-1, n_steps, n_input])  # (batch_size, n_steps, n_input)
    # # permute n_steps and batch_size
    x = tf.transpose(x, [1, 0, 2])
    # # Reshape to prepare input to hidden activation
    x = tf.reshape(x, [-1, n_input])  # (n_steps*batch_size, n_input)
    # # Split data because rnn cell needs a list of inputs for the RNN inner loop
    x = tf.split(0, n_steps, x)  # n_steps * (batch_size, n_input)

    # Define a GRU cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
    # Get lstm cell output
    y, final_state = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
    return y


def var_optimizer(var_list, loss, start_rate=0.1, lrd=True):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = start_rate
    if lrd:
        learning_rate = tf.train.exponential_decay(
            start_rate, global_step, 50, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, return_v = zip(*optimizer.compute_gradients(loss, var_list=var_list))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, return_v), global_step=global_step)
    return optimizer


def var_gradient(var_list, loss, start_rate=0.1, lrd=True):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = start_rate
    if lrd:
        learning_rate = tf.train.exponential_decay(
            start_rate, global_step, 50, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, return_v = zip(*optimizer.compute_gradients(loss, var_list=var_list))
    return gradients


class FitTrendModel(object):
    def __init__(self, input_size, output_size):
        self.graph = tf.Graph()
        self.hyper_cnt = input_size
        self.save_path = "fit_trend.ckpt"
        with self.graph.as_default():
            # 接收输入
            self.ph_hypers = tf.placeholder(tf.float32, shape=[self.hyper_cnt], name='ph_hypers')
            self.tf_hypers, self.reset_vars = assign_diffable_vars2tensor(self.ph_hypers, self.hyper_cnt)
            rnn_step = 5
            trend_input = tf.concat(0, [self.tf_hypers for _ in range(rnn_step)])
            # 通过一个RNN
            trend_outputs = rnn(trend_input)
            print('rnn output')
            print(tf.concat(0, trend_outputs))
            # RNN接一个DNN
            trend_output = dnn(tf.concat(0, trend_outputs), [1, output_size])
            print('dnn output')
            print(trend_output)
            self.predict = trend_output
            # 实际的trend
            self.train_label = tf.placeholder(tf.float32, shape=[output_size], name='train_label')
            # 预测准确率，predict和trend的几何距离
            predict_accuracy = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(trend_output, self.train_label)))) / output_size
            # predict_accuracy /= tf.reduce_mean(tf.concat(0, self.train_label))
            # 稳定时损失，最后一个损失
            stable_loss = tf.unpack(tf.unpack(trend_output)[0])[-1]
            print(stable_loss)
            self.is_fit = tf.placeholder(tf.bool, name='is_fit')
            self.loss = tf.cond(self.is_fit, lambda: predict_accuracy, lambda: stable_loss)

            # 优化器
            self.var_s = tf.trainable_variables()
            self.v_hp_s = self.var_s[0: self.hyper_cnt]
            self.v_fit_s = [v for v in self.var_s if v not in self.v_hp_s]
            self.grads = var_gradient(self.v_hp_s, self.loss, start_rate=0.1, lrd=False)

            def optimize_fit():
                optimizer_fit = var_optimizer(self.v_fit_s, self.loss)
                return optimizer_fit

            def optimize_hp():
                optimizer_hp = var_optimizer(self.v_hp_s, self.loss, start_rate=0.1, lrd=False)
                return optimizer_hp

            self.fit_loss_static = list()
            self.stable_loss_static = list()
            self.optimizer = tf.cond(self.is_fit, optimize_fit, optimize_hp)

            self.saver = tf.train.Saver()

    def init_vars(self, init_hp, session, reset_hp=False):
        init_feed = dict()
        init_feed[self.ph_hypers] = init_hp
        if os.path.exists(self.save_path):
            # Restore variables from disk.
            self.saver.restore(session, self.save_path)
            if reset_hp:
                tf.initialize_variables(var_list=self.reset_vars).run(feed_dict=init_feed)
        else:
            tf.initialize_all_variables().run(feed_dict=init_feed)

    def fit(self, input_data, trend):
        fit_dict = dict()
        fit_dict[self.is_fit] = True
        fit_dict[self.ph_hypers] = input_data
        fit_dict[self.train_label] = trend
        with tf.Session(graph=self.graph) as session:
            self.init_vars(input_data, session)
            _, hps, loss, predict = session.run([self.optimizer, self.tf_hypers, self.loss, self.predict], feed_dict=fit_dict)
            self.saver.save(session, self.save_path)
            self.fit_loss_static.append(loss)
            print('=' * 40)
            print('hps:')
            print(hps)
            print('fit loss:')
            print(loss)
            print('predict:')
            print(predict)
            print('trend')
            print(trend)

    def better_hp(self, input_data, trend):
        fit_dict = dict()
        fit_dict[self.is_fit] = False
        fit_dict[self.ph_hypers] = input_data
        fit_dict[self.train_label] = trend
        with tf.Session(graph=self.graph) as session:
            self.init_vars(input_data, session)
            _, hps, loss, predict, grads = session.run([self.optimizer, self.tf_hypers, self.loss, self.predict, self.grads], feed_dict=fit_dict)
            self.stable_loss_static.append(loss)
            print('*' * 40)
            print('hps:')
            print(hps)
            print('gradients:')
            print(grads)
            print('stable loss:')
            print(loss)
            # print('predict:')
            # print(predict)
            # print('trend')
            # print(trend)

            self.saver.save(session, self.save_path)
            return np.reshape(hps, [self.hyper_cnt])

    def info_collect(self):
        pass

    def dump_collect(self):
        print('stable_loss_static')
        for loss in self.stable_loss_static:
            print(loss)
        print('fit_loss_static')
        for loss in self.fit_loss_static:
            print(loss)


def init_model(input_size, output_size):
    model = FitTrendModel(input_size, output_size)
    return model


def fit_trend(model, input_data, trend):
    model.fit(input_data, trend)


def train_hp(model, input_data, trend):
    return model.better_hp(input_data, trend)
