def test_inputs():
    return [[float(i) for i in range(j , 3 + j )] for j in range(100)]


def test_label(test_data):
    return [hp[0]for hp in test_data]