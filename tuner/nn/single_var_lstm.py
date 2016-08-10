# coding=utf-8

import numpy as np
import tensorflow as tf
import os
import math

# Simple LSTM Model.
from tuner.ctrl.const_define import EMBEDDING_SIZE, batch_cnt_per_step, LINE_FILE_PATH, HP_FILE_PATH, GRAD_FILE_PATH
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
        cnn_raw_hypers = tf.split(0, hyper_cnt, ph_hypers)
        var_init_hypers = [tf.Variable(0.0, name='init_hp_%d' % i, trainable=False) for i in range(hyper_cnt)]
        tf_init_hypers = [var_init_hyper.assign(tf.reshape(cnn_raw_hypers[i], shape=()))
                          for i, var_init_hyper in enumerate(var_init_hypers)]
        var_hypers = [tf.Variable(init_hp, name='hp_%d' % i) for i, init_hp in enumerate(tf_init_hypers)]

        var_reset_hypers = list()
        for var in var_init_hypers:
            var_reset_hypers.append(var)
        for var in var_hypers:
            var_reset_hypers.append(var)

        pack_var_hypers = tf.pack(var_hypers)

        tf_hypers = tf.reshape(pack_var_hypers, [hyper_cnt, EMBEDDING_SIZE])

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

        var_s = tf.trainable_variables()
        print(var_s)
        v_hp = var_s[5: 5 + hyper_cnt]
        # print([v.name for v in v_hp])
        v_fit = list()
        for var in var_s:
            if var not in v_hp:
                v_fit.append(var)

                # print([v.name for v in v_fit])

        optimizer_fit = tf.train.GradientDescentOptimizer(learning_rate)
        gradients_fit, return_v_fit = zip(*optimizer_fit.compute_gradients(loss, var_list=v_fit))
        gradients_fit, _ = tf.clip_by_global_norm(gradients_fit, 1.25)
        # to train hp or not
        optimizer_fit = optimizer_fit.apply_gradients(zip(gradients_fit, return_v_fit), global_step=global_step)

        optimizer_hp = tf.train.GradientDescentOptimizer(learning_rate)
        gradients_hp, return_v_hp = zip(*optimizer_hp.compute_gradients(loss, var_list=v_hp))
        gradients_hp, _ = tf.clip_by_global_norm(gradients_hp, 1.25)

        def hp_opt():
            return optimizer_hp.apply_gradients(zip(gradients_hp, return_v_hp), global_step=global_step)

        optimizer = tf.cond(is_fit, lambda: optimizer_fit, hp_opt)
        # Predictions, not softmax for no label
        train_prediction = logits
        # print(train_prediction)
        saver = tf.train.Saver()

    return graph, saver, is_fit, train_inputs, train_labels, \
           ph_hypers, var_reset_hypers, pack_var_hypers, \
           gradients_hp, optimizer, loss, train_prediction, learning_rate


mean_loss = 0
step = 0
save_path = "model.ckpt"
labels = list()
predicts = list()
mean_loss_vary_cnt = 0


# 每次fit一个step会比较慢
def fit_cnn_loss(input_s, label_s, hyper_s,
                 graph, saver, is_fit,
                 train_inputs, train_labels, ph_hypers, var_reset_hypers, pack_var_hypers,
                 gradients_hp, optimizer, loss, train_prediction, learning_rate,
                 reset=False, train_hyper=False):
    global hyper_cnt
    global batch_size
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
                tf.initialize_variables(var_list=var_reset_hypers).run(feed_dict=init_feed)
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
        grads, _, l, predictions, lr, hyper_f = fit_cnn_ses.run(
            [gradients_hp, optimizer, loss, train_prediction, learning_rate, pack_var_hypers], feed_dict=feed_dict)
        print('fetch_hp:')
        print(hyper_f)
        print('gradients:')
        print(grads)
        mean_loss += l
        # 每次只有第一个loss是有意义的
        # print('label_s')
        # print(label_s)
        labels.append(label_s.reshape(batch_cnt_per_step * (batch_size - hyper_cnt)).tolist()[0])
        predicts.append(predictions.reshape(batch_cnt_per_step * (batch_size - hyper_cnt)).tolist()[0])
        if step % sum_freq == 0:
            if step > 0 and step % (sum_freq * 10) == 0:
                mean_loss /= sum_freq
                # 唯有连续3次损失小于label的5%时才认为可停止
                if mean_loss < np.mean(label_s) * 0.10 and mean_loss < np.mean(predictions) * 0.10:
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
            file_helper.write(LINE_FILE_PATH, '=' * 80)
            print('=' * 80)
            for predict in predicts:
                file_helper.write(LINE_FILE_PATH, str(predict))
                print(predict)
            file_helper.write(LINE_FILE_PATH, '=' * 80)
            del labels[:]
            del predicts[:]
            file_helper.write(LINE_FILE_PATH, str(hyper_s.reshape([hyper_cnt]).tolist()))
        return fit_ret, hyper_f


def train_cnn_hyper(input_s, label_s, init_hypers,
                    graph, saver, is_fit,
                    train_inputs, train_labels, ph_hypers, var_reset_hypers, pack_var_hypers,
                    gradients_hp, optimizer, loss, train_prediction, learning_rate,
                    reset=False):
    # print(train_prediction)
    global hyper_cnt
    hyper_cnt = len(init_hypers)
    sum_freq = 3
    with tf.Session(graph=graph) as fit_cnn_ses:
        if os.path.exists(save_path):
            # Restore variables from disk.
            saver.restore(fit_cnn_ses, save_path)
            if reset:
                print('reset, new hypers:')
                print(init_hypers)
                init_feed = dict()
                init_feed[ph_hypers] = init_hypers
                tf.initialize_variables(var_list=var_reset_hypers).run(feed_dict=init_feed)
                # print("Model restored.")
        else:
            init_feed = dict()
            init_feed[ph_hypers] = init_hypers
            tf.initialize_all_variables().run(feed_dict=init_feed)
            print('Initialized')

        num_step_cnt = 1000
        f_labels = list()
        f_features = list()
        hp_mean_loss = 0
        train_ret = False
        hyper_f = init_hypers
        grads = None
        for step in range(num_step_cnt):
            if train_ret:
                break
            if step == 0:
                hp_input_s = input_s
                hp_label_s = label_s
            else:
                hp_input_s = f_features.pop()
                hp_label_s = f_labels.pop()
            f_features.append(hp_label_s[:, :20, :])
            # print("*" * 80)
            # print(hp_input_s)
            # print(hp_label_s)
            # print("*" * 80)
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
            # print(feed_dict)
            # train
            grads, _, l, predictions, lr, hyper_f = fit_cnn_ses.run(
                [gradients_hp, optimizer, loss, train_prediction, learning_rate, pack_var_hypers], feed_dict=feed_dict)
            f_labels.append(predictions.reshape((batch_cnt_per_step, batch_size - hyper_cnt, EMBEDDING_SIZE)))
            print('fetch_hp:')
            print(hyper_f)
            # tmp_hps = hyper_f.reshape([hyper_cnt]).tolist()
            # file_helper.write(HP_FILE_PATH, str(tmp_hps))
            print('gradients:')
            print(grads)
            hp_mean_loss += l
            if step % sum_freq == 0:
                # print('=' * 35 + 'gradients' + '=' * 35)
                if step > 0:
                    hp_mean_loss /= sum_freq
                print('Average loss at step %d: %f learning rate: %f' % (step, hp_mean_loss, lr))
                # print(hp_s)
                hp_diffs = list()
                better_hp_cnt = 0
                for i in range(hyper_cnt):
                    hp_diffs.append(math.fabs(hyper_f[i] - init_hypers[i]))
                    if hp_diffs[i] > init_hypers[i] * 0.10 and hp_diffs[i] > 0.01:
                        better_hp_cnt += 1
                        if better_hp_cnt >= hyper_cnt / 2:
                            train_ret = True
                            print('=' * 35 + 'hypers' + '=' * 35)
                            print('batch_size, depth, num_hidden, layer_sum, patch_size')
                            print(hyper_f)
                            break

            hp_mean_loss = 0
        final_hps = hyper_f.reshape([hyper_cnt]).tolist()
        file_helper.write(HP_FILE_PATH, str(final_hps))
        file_helper.write(GRAD_FILE_PATH, str(grads))
    return train_ret, hyper_f.tolist()