# coding=utf-8

import numpy as np
import tensorflow as tf
import os

# Simple LSTM Model.
from tuner.ctrl.const_define import EMBEDDING_SIZE, batch_cnt_per_step

num_nodes = 64


def logprob(predictions, labels):
    # prevent negative probability
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


hyper_cnt = 0
batch_size = 20 + hyper_cnt


def init_graph(hypers):
    hyper_cnt = len(hypers)
    batch_size = 20 + hyper_cnt


graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input, Forget, Memory, Output gate: input, previous output, and bias.
    ifcox = tf.Variable(tf.truncated_normal([EMBEDDING_SIZE, num_nodes * 4], -0.1, 1.0))
    ifcom = tf.Variable(tf.truncated_normal([num_nodes, num_nodes * 4], -0.1, 1.0))
    ifcob = tf.Variable(tf.zeros([1, num_nodes * 4]))

    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, EMBEDDING_SIZE], -0.1, 1.0))
    b = tf.Variable(tf.truncated_normal([EMBEDDING_SIZE]))


    def _slice(_x, n, dim):
        return _x[:, n * dim:(n + 1) * dim]


    # Definition of the cell computation.
    def lstm_cell(cur_input, last_output, last_state, drop):
        if drop:
            cur_input = tf.nn.dropout(cur_input, 0.8)
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
    train_inputs = [tf.placeholder(tf.float32, shape=[batch_size - hyper_cnt, EMBEDDING_SIZE]) for _ in
                    range(batch_cnt_per_step)]
    cnn_hypers = tf.placeholder(tf.float32, shape=[hyper_cnt, EMBEDDING_SIZE])

    final_inputs = [tf.concat(0, [data, cnn_hypers]) for data in train_inputs]
    # print(final_inputs)

    train_labels = [tf.placeholder(tf.float32, shape=[batch_size, EMBEDDING_SIZE]) for _ in range(batch_cnt_per_step)]

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    #######################################################################################
    # This is multi lstm layer
    for i in final_inputs:
        output, state = lstm_cell(i, output, state, True)
        outputs.append(output)
    #######################################################################################

    # State saving
    with tf.control_dependencies([saved_output.assign(output),
                                  saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
        print(logits)
        print(tf.concat(0, train_labels))
        loss = tf.reduce_mean(tf.square(tf.sub(logits, tf.concat(0, train_labels))))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        1.0, global_step, 20, 0.6, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)

    # Predictions, not softmax for no label
    train_prediction = logits

    # Sampling and validation eval: batch 1, no unrolling.
    sample_inputs = [tf.placeholder(tf.float32, shape=[batch_size - hyper_cnt, EMBEDDING_SIZE]) for i in
                     range(batch_cnt_per_step)]
    sample_inputs = [tf.concat(0, [sample_input, cnn_hypers]) for sample_input in sample_inputs]
    sample_inputs = tf.concat(0, sample_inputs)
    saved_sample_output = tf.Variable(tf.zeros([batch_size * batch_cnt_per_step, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([batch_size * batch_cnt_per_step, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([batch_size * batch_cnt_per_step, num_nodes])),
        saved_sample_state.assign(tf.zeros([batch_size * batch_cnt_per_step, num_nodes])))
    sample_output, sample_state = lstm_cell(
        sample_inputs, saved_sample_output, saved_sample_state, False)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.xw_plus_b(sample_output, w, b)
    saver = tf.train.Saver()

mean_loss = 0
step = 0


def fit_cnn_loss(input_s, label_s, hyper_s):
    hyper_cnt = len(hyper_s)
    batch_size = 20 + hyper_cnt
    global step
    sum_freq = 5
    labels = []
    predicts = []
    ret = False
    with tf.Session(graph=graph) as fit_cnn_ses:
        if os.path.exists("model.ckpt"):
            # Restore variables from disk.
            saver.restore(fit_cnn_ses, "/tmp/model.ckpt")
            print("Model restored.")
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
        feed_dict[cnn_hypers] = hyper_s
        # train
        _, l, predictions, lr = fit_cnn_ses.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % sum_freq == 0:
            labels.append(label_s.tolist()[0])
            predicts.append(predictions.reshape(batch_cnt_per_step * batch_size).tolist()[0])
            if step > 0 and step % sum_freq * 10 == 0:
                mean_loss /= sum_freq
                # 唯有损失小于label的5%时才认为可停止
                if mean_loss < np.mean(label_s) * 0.05:
                    ret = True
            print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            print('=' * 80)

        # some work for conclusion
        step += 1
        save_path = saver.save(fit_cnn_ses, "model.ckpt")
        print("Model saved in file: %s" % save_path)
        for predict in labels:
            print(predict)
        print('=' * 80)
        for label in predicts:
            print(label)

        # to fetch variables
        feed_dict = dict()
        for i in range(batch_cnt_per_step):
            feed_dict[train_inputs[i]] = input_s[i]
        for i in range(batch_cnt_per_step):
            feed_dict[train_labels[i]] = label_s[i]
        feed_dict[cnn_hypers] = hyper_s
        ifcob_f, ifcom_f, ifcox_f, w_f = fit_cnn_ses.run([ifcob, ifcom, ifcox, w, b], feed_dict=feed_dict)
        if ret:
            return ifcob_f, ifcom_f, ifcox_f, w_f
        else:
            return None, None, None, None


def train_cnn_hyper(ifcob_f, ifcom_f, ifcox_f, w_f, init_input, init_label, hyper_s):
    # 优化超参
    hp_graph = tf.Graph()
    with hp_graph.as_default():
        # Parameters:
        # Input, Forget, Memory, Output gate: input, previous output, and bias.
        hp_ifcox = tf.placeholder(tf.float32, shape=[EMBEDDING_SIZE, num_nodes * 4])
        hp_ifcom = tf.placeholder(tf.float32, shape=[num_nodes, num_nodes * 4])
        hp_ifcob = tf.placeholder(tf.float32, shape=[1, num_nodes * 4])

        # Variables saving state across unrollings.
        hp_saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        hp_saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        hp_w = tf.placeholder(tf.float32, shape=[num_nodes, EMBEDDING_SIZE])

        # Definition of the cell computation.
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

        # Input data.
        hp_train_inputs = [tf.placeholder(tf.float32, shape=[batch_size - hyper_cnt, EMBEDDING_SIZE]) for _ in
                           range(batch_cnt_per_step)]
        # 归一化
        hp_cnn_hypers = [tf.Variable(float(hyper_s[0][0]) / 10.0).value(),
                         tf.Variable(float(hyper_s[1][0]) / 10.0).value(),
                         tf.Variable(float(hyper_s[2][0]) / 10.0).value(),
                         tf.Variable(float(hyper_s[3][0]) / 10.0).value()]
        hp_cnn_hypers = [tf.mul(hp_cnn_hyper, 10.0) for hp_cnn_hyper in hp_cnn_hypers]
        hp_cnn_hypers = tf.reshape(tf.pack(hp_cnn_hypers), [hyper_cnt, EMBEDDING_SIZE])

        hp_final_inputs = [tf.concat(0, [data, hp_cnn_hypers]) for data in hp_train_inputs]
        # print(final_inputs)

        hp_train_labels = [tf.placeholder(tf.float32, shape=[batch_size, EMBEDDING_SIZE]) for _ in
                           range(batch_cnt_per_step)]

        # Unrolled LSTM loop.
        hp_outputs = list()
        hp_output = hp_saved_output
        hp_state = hp_saved_state
        #######################################################################################
        # This is multi lstm layer
        for i in hp_final_inputs:
            hp_output, state = hp_lstm_cell(i, hp_output, hp_state, True)
            hp_outputs.append(hp_output)
        #######################################################################################

        # State saving
        with tf.control_dependencies([hp_saved_output.assign(hp_output),
                                      hp_saved_state.assign(hp_state)]):
            # Classifier.
            hp_logits = tf.matmul(tf.concat(0, hp_outputs), hp_w)
            print(hp_logits)
            print(tf.concat(0, hp_train_labels))
            # hp_loss = tf.split(0, batch_size * batch_cnt_per_step, hp_logits)[batch_size * batch_cnt_per_step - 1]
            hp_loss = tf.reduce_mean(tf.square(hp_logits))

        # Optimizer.
        hp_global_step = tf.Variable(0, trainable=False)
        hp_learning_rate = tf.train.exponential_decay(
            0.1, hp_global_step, 20, 0.1, staircase=True)

        hp_optimizer = tf.train.GradientDescentOptimizer(hp_learning_rate)
        hp_gradients, v = zip(*hp_optimizer.compute_gradients(hp_loss))
        print(hp_gradients)
        hp_gradients, _ = tf.clip_by_global_norm(hp_gradients, 1.25)
        return_gradients = tf.pack(hp_gradients)
        hp_optimizer = hp_optimizer.apply_gradients(
            zip(hp_gradients, v), global_step=hp_global_step)
        # Predictions, not softmax for no label
        hp_train_prediction = hp_logits

    hp_num_steps = 800
    hp_sum_freq = 50
    hp_loss_es = []
    f_labels = list()
    f_features = list()

    with tf.Session(graph=hp_graph) as session:
        tf.initialize_all_variables().run()
        # train_batch.reset()
        print('Initialized')
        hp_mean_loss = 0
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
            # print("*" * 80)
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
            _, hp_s, grads, l, lr, f_pred = session.run(
                [hp_optimizer, hp_cnn_hypers, return_gradients, hp_loss, hp_learning_rate, hp_train_prediction], feed_dict=hp_feed_dict)
            # print("hp_train_prediction shape:")
            # print(f_pred.shape)
            f_labels.append(f_pred.reshape((batch_cnt_per_step, batch_size, EMBEDDING_SIZE)))
            hp_mean_loss += l
            #     hp_predicts.append(f_prediction.reshape(batch_cnt_per_step * batch_size).tolist()[0])
            if step % hp_sum_freq == 0:
                print('=' * 35 + 'gradients' + '=' * 35)
                print(grads)
                hp_loss_es.append(l)
                # print(predictions.reshape(batch_cnt_per_step * (batch_size - hyper_cnt)).tolist())
                # print(label_s)
                if step > 0:
                    hp_mean_loss /= hp_sum_freq
                # The mean loss is an estimate of the loss over the last few batches.
                print(
                    'Average loss at step %d: %f learning rate: %f' % (step, hp_mean_loss, lr))
                hp_mean_loss = 0

                print('hypers:')
                print(hp_s)
                print('=' * 80)

        print(hp_loss_es)
