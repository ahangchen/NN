from tuner.nn.hp2trend import init_model, fit_trend, train_hp
import numpy as np

model = None


def fit(input_data, trend):
    global model
    if model is None:
        model = init_model(input_data, trend)
    fit_trend(model, np.array(input_data), np.array(trend))


def train(input_data, trend):
    global model
    return train_hp(model, np.array(input_data), np.array(trend))


def trend2_better_hp(input_data, trend):
    fit(input_data, trend)
    return train(input_data, trend)