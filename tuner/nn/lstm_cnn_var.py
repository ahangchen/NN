# coding=utf-8

import numpy as np
import tensorflow as tf
import os
import math

# Simple LSTM Model.
from tuner.ctrl.const_define import EMBEDDING_SIZE, batch_cnt_per_step, LINE_FILE_PATH, HP_FILE_PATH
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
        ifcox = tf.Variable(tf.truncated_normal([EMBEDDING_SIZE, num_nodes * 4], mean=0.0, stddev=0.1))
        ifcom = tf.Variable(tf.truncated_normal([num_nodes, num_nodes * 4], mean=0.0, stddev=0.1))
        ifcob = tf.Variable(tf.zeros([1, num_nodes * 4]))

        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size - hyper_cnt, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size - hyper_cnt, num_nodes]), trainable=False)
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, EMBEDDING_SIZE], mean=0.0, stddev=0.1))
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
        ph_hypers = tf.placeholder(tf.float32, shape=[hyper_cnt, EMBEDDING_SIZE], name='ph_hypers')

        concat_inputs = [tf.concat(0, [data, ph_hypers]) for data in train_inputs]
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
            loss = tf.reduce_mean(tf.sqrt(tf.square(tf.sub(logits, tf.concat(0, train_labels)))))

        # Optimizer.
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            0.5, global_step, 50, 0.8, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        print(v[0])
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)

        # Predictions, not softmax for no label
        train_prediction = logits
        saver = tf.train.Saver()
    return graph, saver, train_inputs, train_labels, ph_hypers, optimizer, loss, train_prediction, learning_rate, \
           ifcob, ifcom, ifcox, w, b, wi, bi


mean_loss = 0
step = 0
save_path = "model.ckpt"
labels = list()
predicts = list()
mean_loss_vary_cnt = 0


# 每次fit一个step会比较慢
def fit_cnn_loss(input_s, label_s, hyper_s,
                 graph, saver,
                 train_inputs, train_labels, ph_hypers,
                 optimizer, loss, train_prediction, learning_rate,
                 ifcob, ifcom, ifcox, w, b, wi, bi,
                 reset=False):
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
    # todo only for test
    ret = True
    with tf.Session(graph=graph) as fit_cnn_ses:
        if os.path.exists(save_path):
            # Restore variables from disk.
            saver.restore(fit_cnn_ses, save_path)
            # print("Model restored.")
        else:
            tf.initialize_all_variables().run()
            print('Initialized')

        global mean_loss
        mean_loss = 0
        # prepare and feed train data
        feed_dict = dict()
        for i in range(batch_cnt_per_step):
            feed_dict[train_inputs[i]] = input_s[i]
        for i in range(batch_cnt_per_step):
            feed_dict[train_labels[i]] = label_s[i]
        feed_dict[ph_hypers] = hyper_s
        # train
        _, l, predictions, lr = fit_cnn_ses.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        # 每次只有第一个loss是有意义的
        # print('label_s')
        # print(label_s.tolist())
        labels.append(label_s.reshape(batch_cnt_per_step * (batch_size - hyper_cnt)).tolist()[0])
        predicts.append(predictions.reshape(batch_cnt_per_step * (batch_size - hyper_cnt)).tolist()[0])
        # print('step: %d' % step)
        if step % sum_freq == 0:
            if step > 0 and step % (sum_freq * 10) == 0:
                mean_loss /= sum_freq
                # 唯有连续3次损失小于label的5%时才认为可停止
                if mean_loss < np.mean(label_s) * 0.10 and mean_loss < np.mean(predictions) * 0.10:
                    mean_loss_vary_cnt += 1
                else:
                    mean_loss_vary_cnt = 0
                if mean_loss_vary_cnt >= 5:
                    ret = True
                    print('mean loss < label_s * 10%')
                print(mean_loss)
                print(np.mean(label_s))
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0

        # some work for conclusion
        step += 1
        saver.save(fit_cnn_ses, save_path)
        # print("Model saved in file: %s" % save_path)
        if ret:
            for label in labels:
                file_helper.write(LINE_FILE_PATH, str(label))
                # print(label)
            file_helper.write(LINE_FILE_PATH, '=' * 80)
            # print('=' * 80)
            for predict in predicts:
                file_helper.write(LINE_FILE_PATH, str(predict))
                # print(predict)
            file_helper.write(LINE_FILE_PATH, '=' * 80)
            del labels[:]
            del predicts[:]
            file_helper.write(LINE_FILE_PATH, str(hyper_s.reshape([hyper_cnt]).tolist()))
            ifcob_f, ifcom_f, ifcox_f, w_f, b_f, wi_f, bi_f = fit_cnn_ses.run([ifcob, ifcom, ifcox, w, b, wi, bi], feed_dict=feed_dict)
            return ifcob_f, ifcom_f, ifcox_f, w_f, b_f, wi_f, bi_f
        else:
            return None, None, None, None, None, None, None


def train_cnn_hyper(ifcob_f, ifcom_f, ifcox_f, w_f, b_f, wi_f, bi_f, init_input, init_label, var_init_hypers):
    hp_graph = tf.Graph()
    with hp_graph.as_default():
        hp_ifcox = tf.placeholder(tf.float32, shape=[EMBEDDING_SIZE, num_nodes * 4], name='hp_ifcox')
        hp_ifcom = tf.placeholder(tf.float32, shape=[num_nodes, num_nodes * 4], name='hp_ifcom')
        hp_ifcob = tf.placeholder(tf.float32, shape=[1, num_nodes * 4], name='hp_ifcob')

        hp_saved_output = tf.Variable(tf.zeros([batch_size - hyper_cnt, num_nodes]), trainable=False)
        hp_saved_state = tf.Variable(tf.zeros([batch_size - hyper_cnt, num_nodes]), trainable=False)
        hp_w = tf.placeholder(tf.float32, shape=[num_nodes, EMBEDDING_SIZE], name='hp_w')
        hp_b = tf.placeholder(tf.float32, shape=[EMBEDDING_SIZE], name='hp_b')

        def hp_lstm_cell(cur_input, last_output, last_state, drop):
            if drop:
                cur_input = tf.nn.dropout(cur_input, 0.8)
            ifco_gates = tf.matmul(cur_input, hp_ifcox) + tf.matmul(last_output, hp_ifcom) + hp_ifcob
            input_gate = tf.sigmoid(_slice(ifco_gates, 0, num_nodes))
            forget_gate = tf.sigmoid(_slice(ifco_gates, 1, num_nodes))
            update = _slice(ifco_gates, 2, num_nodes)
            last_state = forget_gate * last_state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(_slice(ifco_gates, 3, num_nodes))
            output_gate *= tf.tanh(last_state)
            if drop:
                output_gate = tf.nn.dropout(output_gate, 0.8)
            return output_gate, last_state

        hp_train_inputs = [tf.placeholder(tf.float32, shape=[batch_size - hyper_cnt, EMBEDDING_SIZE],
                                          name='hp_train_inputs{i}'.format(i=i)) for i in range(batch_cnt_per_step)]
        pack_var_hypers = [tf.Variable(var_init_hypers[i][0]).value() for i in range(hyper_cnt)]
        tf_hypers = tf.reshape(tf.pack(pack_var_hypers), [hyper_cnt, EMBEDDING_SIZE])

        hp_concat_inputs = [tf.concat(0, [data, tf_hypers]) for data in hp_train_inputs]
        hp_wi = tf.placeholder(tf.float32, shape=[batch_size, batch_size - hyper_cnt], name='hp_wi')
        hp_bi = tf.placeholder(tf.float32, shape=[batch_size - hyper_cnt], name='hp_bi')
        hp_final_inputs = tf.nn.xw_plus_b(tf.reshape(tf.concat(0, hp_concat_inputs), [batch_cnt_per_step, batch_size]),
                                          hp_wi, hp_bi)
        hp_final_inputs = tf.split(0, batch_cnt_per_step, hp_final_inputs)
        hp_train_labels = [tf.placeholder(tf.float32, shape=[batch_size - hyper_cnt, EMBEDDING_SIZE],
                                          name='hp_train_labels{i}'.format(i=i)) for i in range(batch_cnt_per_step)]

        hp_outputs = list()
        hp_output = hp_saved_output
        hp_state = hp_saved_state
        #######################################################################################
        # This is multi lstm layer
        for hp_final_input in hp_final_inputs:
            hp_final_input = tf.reshape(hp_final_input, [batch_size - hyper_cnt, EMBEDDING_SIZE])
            hp_output, state = hp_lstm_cell(hp_final_input, hp_output, hp_state, True)
            hp_outputs.append(hp_output)
        #######################################################################################

        with tf.control_dependencies([hp_saved_output.assign(hp_output),
                                      hp_saved_state.assign(hp_state)]):
            hp_logits = tf.nn.xw_plus_b(tf.concat(0, hp_outputs), hp_w, hp_b)
            # hp_loss = tf.reduce_mean(tf.square(hp_logits))
            # for the max change between losses
            hp_loss = tf.reduce_mean(tf.sub(hp_logits,
                                            tf.reshape(
                                                tf.concat(
                                                    0, hp_final_inputs
                                                ),
                                                [batch_cnt_per_step * (batch_size - hyper_cnt), EMBEDDING_SIZE]
                                            )))

        hp_global_step = tf.Variable(0, trainable=False)
        hp_learning_rate = tf.train.exponential_decay(
            0.001, hp_global_step, 100, 0.9, staircase=True)

        hp_optimizer = tf.train.GradientDescentOptimizer(hp_learning_rate)
        hp_gradients, v = zip(*hp_optimizer.compute_gradients(hp_loss))
        hp_gradients, _ = tf.clip_by_global_norm(hp_gradients, 1.25)
        return_gradients = tf.pack(hp_gradients)
        hp_optimizer = hp_optimizer.apply_gradients(
            zip(hp_gradients, v), global_step=hp_global_step)
        hp_train_prediction = hp_logits
    # todo for test
    hp_num_steps = 1
    hp_sum_freq = 50
    hp_loss_es = []
    f_labels = list()
    f_features = list()
    mean_loss_collect = list()

    with tf.Session(graph=hp_graph) as session:
        tf.initialize_all_variables().run()
        # print('Initialized')
        hp_mean_loss = 0
        hp_s = []
        ret = False
        for step in range(hp_num_steps):
            if step == 0:
                hp_input_s = init_input
                hp_label_s = init_label
            else:
                hp_input_s = f_features.pop()
                hp_label_s = f_labels.pop()
            f_features.append(hp_label_s[:, :20, :])
            # print("*" * 80)
            # print(hp_input_s)
            # print(hp_label_s)
            # print("*" * 80)
            hp_feed_dict = dict()
            for i in range(batch_cnt_per_step):
                hp_feed_dict[hp_train_inputs[i]] = hp_input_s[i]
            for i in range(batch_cnt_per_step):
                hp_feed_dict[hp_train_labels[i]] = hp_label_s[i]
            hp_feed_dict[hp_ifcob] = ifcob_f
            hp_feed_dict[hp_ifcom] = ifcom_f
            hp_feed_dict[hp_ifcox] = ifcox_f
            hp_feed_dict[hp_w] = w_f
            hp_feed_dict[hp_b] = b_f
            hp_feed_dict[hp_wi] = wi_f
            hp_feed_dict[hp_bi] = bi_f
            _, hp_s, grads, l, lr, f_pred = session.run(
                [hp_optimizer, tf_hypers, return_gradients, hp_loss, hp_learning_rate, hp_train_prediction],
                feed_dict=hp_feed_dict)
            f_labels.append(f_pred.reshape((batch_cnt_per_step, batch_size - hyper_cnt, EMBEDDING_SIZE)))
            hp_mean_loss += l
            mean_loss_collect.append(l)
            print('grads:')
            print(grads)
            if step % hp_sum_freq == 0:
                # print('=' * 35 + 'gradients' + '=' * 35)
                hp_loss_es.append(l)
                if step > 0:
                    hp_mean_loss /= hp_sum_freq
                # print('Average loss at step %d: %f learning rate: %f' % (step, hp_mean_loss, lr))
                # print(hp_s)
                hp_diffs = list()
                better_hp_cnt = 0
                for i in range(hyper_cnt):
                    hp_diffs.append(math.fabs(hp_s[i][0] - var_init_hypers[i][0]))
                    if hp_diffs[i] > var_init_hypers[i][0] * 0.20 and hp_diffs[i] > 0.01:
                        print('var_init_hypers[i][0]: ')
                        print(var_init_hypers[i][0])
                        better_hp_cnt += 1
                        if better_hp_cnt >= hyper_cnt / 2:
                            ret = True
                            print('=' * 35 + 'hypers' + '=' * 35)
                            print('batch_size, depth, num_hidden, layer_sum, patch_size')
                            print(hp_s)
                            break

                if ret:
                    # print('hp_diffs:')
                    # print (hp_diffs)
                    break
            hp_mean_loss = 0
    final_hps = hp_s.reshape([hyper_cnt]).tolist()
    file_helper.write(HP_FILE_PATH, str(final_hps))
    return ret, final_hps
