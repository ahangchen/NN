# coding=utf-8
import random

import numpy as np
import tensorflow as tf
import os
import math
from scipy.optimize import leastsq

# Simple LSTM Model.
from tuner.ctrl.const_define import EMBEDDING_SIZE, batch_cnt_per_step, LINE_FILE_PATH, PREDICT_FILE_PATH
from tuner.util import file_helper

num_nodes = 64


def logprob(predictions, labels):
    # prevent negative probability
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


hyper_cnt = 0
batch_size = 0
has_init = False


def init_graph(hypers, cnn_batch_size):
    global hyper_cnt
    global batch_size
    global has_init
    hyper_cnt = len(hypers)
    batch_size = cnn_batch_size + hyper_cnt
    return init_model()


def _slice(_x, n, dim):
    return _x[:, n * dim:(n + 1) * dim]

init_norm = False
norm_list = list()


def norm(params):
    global init_norm
    if not init_norm:
        print('init norm')
        for param in params:
            norm_list.append(param * 10.0)
        print(norm_list)
        init_norm = True


def init_model():
    global hyper_cnt
    global batch_size
    graph = tf.Graph()
    with graph.as_default():
        # Parameters:
        # Input, Forget, Memory, Output gate: input, previous output, and bias.
        is_fit = tf.placeholder(tf.bool)
        ifcox = tf.Variable(tf.truncated_normal([EMBEDDING_SIZE, num_nodes * 4], mean=-0.1, stddev=0.1))
        ifcom = tf.Variable(tf.truncated_normal([num_nodes, num_nodes * 4], mean=-0.1, stddev=0.1))
        ifcob = tf.Variable(tf.zeros([1, num_nodes * 4]))

        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size - hyper_cnt, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size - hyper_cnt, num_nodes]), trainable=False)
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, EMBEDDING_SIZE], mean=-0.1, stddev=0.1))
        b = tf.Variable(tf.truncated_normal([EMBEDDING_SIZE]))

        # Definition of the cell computation.
        def lstm_cell(cur_input, last_output, last_state, drop):
            if drop:
                cur_input = tf.nn.dropout(cur_input, 0.8)
            with tf.device('/cpu:0'):
                ifco_gates = tf.matmul(cur_input, ifcox) + tf.matmul(last_output, ifcom) + ifcob
                input_gate = tf.sigmoid(_slice(ifco_gates, 0, num_nodes))
                forget_gate = tf.sigmoid(_slice(ifco_gates, 1, num_nodes))
                update = _slice(ifco_gates, 2, num_nodes)
                last_state = forget_gate * last_state + input_gate * tf.tanh(update)
                output_gate = tf.sigmoid(_slice(ifco_gates, 3, num_nodes))
                output_gate *= tf.tanh(last_state)
            if drop:
                output_gate = tf.nn.dropout(output_gate, 0.8)
            return output_gate, last_state

        # Input data.
        train_inputs = [tf.placeholder(tf.float32, shape=[batch_size - hyper_cnt, EMBEDDING_SIZE],
                                       name='train_input{i}'.format(i=i)) for i in range(batch_cnt_per_step)]
        # when fit, feed placeholder to variable, not train for variable,
        # when not fit, train variable, feed to placeholder, feed placeholder to variable
        ph_hypers = tf.placeholder(tf.float32, shape=[hyper_cnt], name='ph_hypers')

        tf_hypers = tf.reshape(ph_hypers, [hyper_cnt, EMBEDDING_SIZE])

        concat_inputs = [tf.concat(0, [data, tf_hypers]) for data in train_inputs]
        wi = tf.Variable(tf.truncated_normal([batch_size, batch_size - hyper_cnt], mean=-0.1, stddev=0.1))
        bi = tf.Variable(tf.truncated_normal([batch_size - hyper_cnt]))
        final_inputs = tf.nn.xw_plus_b(tf.reshape(tf.concat(0, concat_inputs), [batch_cnt_per_step, batch_size]), wi,
                                       bi)
        final_inputs = tf.split(0, batch_cnt_per_step, final_inputs)
        train_labels = [tf.placeholder(tf.float32, shape=[batch_size - hyper_cnt, EMBEDDING_SIZE],
                                       name='train_labels{i}'.format(i=i)) for i in range(batch_cnt_per_step)]

        # Unrolled LSTM loop.
        outputs = list()
        output = saved_output
        state = saved_state
        #######################################################################################
        # This is multi lstm layer
        for final_input in final_inputs:
            final_input = tf.reshape(final_input, [batch_size - hyper_cnt, EMBEDDING_SIZE])
            output, state = lstm_cell(final_input, output, state, False)
            outputs.append(output)
        #######################################################################################

        # State saving
        with tf.control_dependencies([saved_output.assign(output),
                                      saved_state.assign(state)]):
            # Classifier.
            logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            # print(logits)
            # print(tf.concat(0, train_labels))
            loss = tf.cond(is_fit,
                           lambda: tf.reduce_mean(tf.sqrt(tf.square(tf.sub(logits, tf.concat(0, train_labels))))),
                           lambda: tf.reduce_mean(logits))

        # Optimizer.
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            0.5, global_step, 50, 0.8, staircase=True)

        optimizer_fit = tf.train.GradientDescentOptimizer(learning_rate)
        gradients_fit, return_v_fit = zip(*optimizer_fit.compute_gradients(loss, tf.trainable_variables()))
        gradients_fit, _ = tf.clip_by_global_norm(gradients_fit, 1.25)
        # to train hp or not
        optimizer_fit = optimizer_fit.apply_gradients(zip(gradients_fit, return_v_fit), global_step=global_step)

        # Predictions, not softmax for no label
        train_prediction = logits
        # print(train_prediction)
        saver = tf.train.Saver()

    return graph, saver, is_fit, train_inputs, train_labels, \
           ph_hypers, optimizer_fit, loss, train_prediction, learning_rate


mean_loss = 0
step = 0
save_path = "model.ckpt"
labels = list()
predicts = list()
mean_loss_vary_cnt = 0


# 每次fit一个step会比较慢
def fit_cnn_loss(input_s, label_s, hyper_s,
                 graph, saver, is_fit,
                 train_inputs, train_labels, ph_hypers,
                 optimizer, loss, train_prediction, learning_rate,
                 reset=False, train_hyper=False):
    global hyper_cnt
    global batch_size
    norm(hyper_s)
    hyper_s = [hyper / norm_list[i] for i, hyper in enumerate(hyper_s)]
    hyper_cnt = len(hyper_s)
    global step
    global save_path
    sum_freq = 3
    global labels
    global predicts
    global mean_loss_vary_cnt
    if reset == 1:
        step = 0
    fit_ret = False
    with tf.Session(graph=graph) as fit_cnn_ses:
        if os.path.exists(save_path):
            # Restore variables from disk.
            saver.restore(fit_cnn_ses, save_path)
            if reset:
                print('reset, new hypers:')
                print(hyper_s)
                init_feed = dict()
                init_feed[ph_hypers] = hyper_s
                tf.initialize_all_variables().run(feed_dict=init_feed)
                # print("Model restored.")
        else:
            init_feed = dict()
            init_feed[ph_hypers] = hyper_s
            tf.initialize_all_variables().run(feed_dict=init_feed)
            print('Initialized')

        global mean_loss
        mean_loss = 0
        # prepare and feed train data
        feed_dict = dict()
        feed_dict[is_fit] = train_hyper
        for i in range(batch_cnt_per_step):
            feed_dict[train_inputs[i]] = input_s[i]
        for i in range(batch_cnt_per_step):
            feed_dict[train_labels[i]] = label_s[i]
        feed_dict[ph_hypers] = hyper_s
        # print(feed_dict)
        # train
        _, l, predictions, lr = fit_cnn_ses.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        # 每次只有第一个loss是有意义的
        # print('label_s')
        # print(label_s)
        labels.append(label_s.reshape(batch_cnt_per_step * (batch_size - hyper_cnt)).tolist()[0])
        predicts.append(predictions.reshape(batch_cnt_per_step * (batch_size - hyper_cnt)).tolist()[0])
        if step % sum_freq == 0:
            # 次数为奇偶时梯度呈现翻转，所以不能固定在偶数时验证
            fit_verify = random.randint(9, 10)
            if step > 0 and step % (sum_freq * fit_verify) == 0:
                mean_loss /= sum_freq
                # 唯有连续3次损失小于label的5%时才认为可停止
                if mean_loss < np.mean(label_s) * 0.15 and mean_loss < np.mean(predictions) * 0.15:
                    mean_loss_vary_cnt += 1
                else:
                    mean_loss_vary_cnt = 0
                if mean_loss_vary_cnt >= 5:
                    fit_ret = True
                    print('mean loss < label_s * 10%')
                print(mean_loss)
                print(np.mean(label_s))
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0

        # some work for conclusion
        step += 1
        saver.save(fit_cnn_ses, save_path)
        # print("Model saved in file: %s" % save_path)
        if fit_ret:
            for label in labels:
                file_helper.write(LINE_FILE_PATH, str(label))
                print(label)
            print('=' * 80)
            for predict in predicts:
                file_helper.write(PREDICT_FILE_PATH, str(predict))
                print(predict)
            del labels[:]
            del predicts[:]
        return fit_ret


def predict_loss(input_s, label_s, init_hypers,
                 graph, saver, is_fit,
                 train_inputs, train_labels, ph_hypers,
                 optimizer, loss, train_prediction, learning_rate,
                 reset=False):
    init_hypers = [hyper / norm_list[i] for i, hyper in enumerate(init_hypers)]
    with tf.Session(graph=graph) as fit_cnn_ses:
        if os.path.exists(save_path):
            # Restore variables from disk.
            saver.restore(fit_cnn_ses, save_path)
            if reset:
                print('reset, new hypers:')
                print(init_hypers)
                init_feed = dict()
                init_feed[ph_hypers] = init_hypers
                tf.initialize_all_variables().run(feed_dict=init_feed)
        else:
            init_feed = dict()
            init_feed[ph_hypers] = init_hypers
            tf.initialize_all_variables().run(feed_dict=init_feed)
            print('Initialized')

        cur_idx = 0
        f_labels = list()
        f_features = list()
        log_labels = list()
        end_train = False
        x_s = np.array([float(i) for i in range(batch_size - hyper_cnt)])
        while True:
            if end_train:
                break
            if cur_idx == 0:
                hp_input_s = input_s
                hp_label_s = label_s
            else:
                hp_input_s = f_features.pop()
                hp_label_s = f_labels.pop()
            f_features.append(hp_label_s[:, :20, :])
            feed_dict = dict()
            for i in range(batch_cnt_per_step):
                feed_dict[train_inputs[i]] = hp_input_s[i]
            for i in range(batch_cnt_per_step):
                feed_dict[train_labels[i]] = hp_label_s[i]
            feed_dict = dict()
            feed_dict[is_fit] = False
            for i in range(batch_cnt_per_step):
                feed_dict[train_inputs[i]] = input_s[i]
            for i in range(batch_cnt_per_step):
                feed_dict[train_labels[i]] = label_s[i]
            feed_dict[ph_hypers] = init_hypers
            _, l, predictions, lr = fit_cnn_ses.run(
                [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
            f_labels.append(predictions.reshape((batch_cnt_per_step, batch_size - hyper_cnt, EMBEDDING_SIZE)))

            def residuals(p, x, y):
                return p[0] * x + p[1] - y

            p0 = [-1.0, 1.0]
            predict_losses = predictions.reshape([batch_cnt_per_step, batch_size - hyper_cnt])
            plsq = leastsq(residuals, p0, args=(x_s, predict_losses[-1]))
            k = math.fabs(plsq[0][0])
            print(k)

            cur_idx += 1
            if k < 0.1 and k < np.mean(predict_losses[-1]):
                end_train = True
                for predict in predictions.reshape((batch_cnt_per_step * (batch_size - hyper_cnt))).tolist():
                    log_labels.append(predict)
            else:
                log_labels.append(predict_losses[0][0])
        for predict in log_labels:
            file_helper.write(PREDICT_FILE_PATH, str(predict))
        file_helper.write(PREDICT_FILE_PATH, '===')
    # 返回结果与预测次数
    return end_train, cur_idx
