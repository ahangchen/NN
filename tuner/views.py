# coding=utf-8
import json

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from tuner.nn.tf_train import train_cnn, CONTINUE_TRAIN, END_TRAIN, NN_OK, better_hyper
from tuner.util import json_helper


@csrf_exempt
def train(request):
    """
    传入loss和当前hyper,task id，返回是否收敛（实际上是上一次是否收敛）
    需要monster在每次训练fetch loss和hyper，并判断是否收敛，收敛则停止训练，并调用hyper
    :param request:
    :return:
    """
    cur_loss = request.POST.getlist('loss')
    cur_loss = json.loads(cur_loss[0])
    hypers = request.POST.getlist('hyper')
    hypers = json.loads(hypers[0])
    reset = int(request.POST['reset'])
    ret = train_cnn(reset, hypers, cur_loss[: 120])
    if CONTINUE_TRAIN == ret:
        return json_helper.dump_err_msg(CONTINUE_TRAIN, "haven't fit cnn loss")
    elif END_TRAIN == ret:
        return json_helper.dump_err_msg(END_TRAIN, "has fit cnn loss")


@csrf_exempt
def hyper(request):
    """
    传入task id，返回新的hyper
    重新训练monster神经网络，并不断调用train
    :param request:
    :return:
    """
    cur_loss = request.POST.getlist('loss')
    cur_loss = json.loads(cur_loss[0])
    hypers = request.POST.getlist('hyper')
    hypers = json.loads(hypers[0])
    return json_helper.dump_err_msg(NN_OK, better_hyper(cur_loss, hypers))
