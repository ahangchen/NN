import json

from django.http import HttpResponse


def dump_err_msg(err_code, err_msg):
    return HttpResponse(json.dumps({'ret': err_code, 'msg': err_msg}))
