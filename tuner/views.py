from django.http import HttpResponse


# Create your views here.


def train(request):
    """
    传入loss和当前hyper,task id，返回是否收敛（实际上是上一次是否收敛）
    需要monster在每次训练fetch loss和hyper，并判断是否收敛，收敛则停止训练，并调用hyper
    :param request:
    :return:
    """
    return HttpResponse(request)


def hyper(request):
    """
    传入task id，返回新的hyper
    重新训练monster神经网络，并不断调用train
    :param request:
    :return:
    """
    return HttpResponse(request)
