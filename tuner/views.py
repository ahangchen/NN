# coding=utf-8
import json

# Create your views here.
from tuner.nn.tf_train import train_cnn, CONTINUE_TRAIN, END_TRAIN, NN_OK, best_hyper
from tuner.util import json_helper


def train(request):
    """
    传入loss和当前hyper,task id，返回是否收敛（实际上是上一次是否收敛）
    需要monster在每次训练fetch loss和hyper，并判断是否收敛，收敛则停止训练，并调用hyper
    :param request:
    :return:
    """
    cur_loss = request.POST['cur_loss']
    hyper_json = request.POST['hyper_json']
    hypers = json.loads(hyper_json)
    ret = train_cnn(hypers, cur_loss)
    if CONTINUE_TRAIN == ret:
        return json_helper.dump_err_msg(CONTINUE_TRAIN, '未收敛，继续训练网络参数')
    elif END_TRAIN == ret:
        return json_helper.dump_err_msg(END_TRAIN, '已收敛，开始优化超参')


def hyper(request):
    """
    传入task id，返回新的hyper
    重新训练monster神经网络，并不断调用train
    :param request:
    :return:
    """
    return json_helper.dump_err_msg(NN_OK, best_hyper())
