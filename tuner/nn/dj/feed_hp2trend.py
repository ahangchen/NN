from tuner.nn.hp2trend import init_model, fit_trend, train_hp
import numpy as np

model = None
cur_trends = list()


def fit(input_data, trend):
    # print(trend)
    global model
    if model is None:
        model = init_model(len(input_data), len(trend))
    fit_trend(model, np.array(input_data), np.array(trend))


def train(input_data, trend):
    global model
    return train_hp(model, np.array(input_data), np.array(trend))


def trend2_better_hp(input_data, trend):
    cur_trends.append(trend)
    cnt = 20 - len(cur_trends)
    for _ in range(cnt):
        print(trend)
        fit(input_data, trend)
    for trend_i in cur_trends:
        print(trend_i)
        fit(input_data, trend_i)
    if len(cur_trends) >= 20:
        cur_trends.pop(0)
    # for _ in range(5):
    #     train(input_data, trend)
    return train(input_data, trend)