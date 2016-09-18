from tuner.nn.hp2trend import init_model, fit_trend
import numpy as np
import random


def test_inputs(hp_cnt):
    test_data = [[random.uniform(0, 1) + 0.1 for _ in range(hp_cnt)] for _ in range(1, 101)]
    random.shuffle(test_data)
    return test_data


def test_label(test_data):
    return [[hp[0] / (x * hp[1] * 10.0 + hp[2] * hp[3]) * hp[4] for x in range(1, 10)] for hp in test_data]


if __name__ == '__main__':
    test_hps = test_inputs(5)
    test_labels = test_label(test_hps)
    for test_hp in test_hps:
        test_hp.extend(range(1, 10))
    print('test_data:')
    print(len(test_hps))
    print('test_label:')
    print(len(test_labels))
    model = init_model(len(test_hps[0]), len(test_labels[0]))
    for _ in range(20):
        for test_hp, test_label in zip(test_hps, test_labels):
            fit_trend(model, np.array(test_hp), np.array(test_label))


