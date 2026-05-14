import numpy as np
from scipy.io.arff import loadarff

file_path = './dataset/'
train_fn = "FordA_TRAIN.arff"
test_fn = "FordA_TEST.arff"


def _read_arff(path):
    raw, meta = loadarff(path)
    cols = list(meta)
    data2d = np.zeros([raw.shape[0], len(cols)])
    for i, col in enumerate(cols):
        data2d[:, i] = raw[col]
    return data2d


def _split_xy(data):
    # FordA: 마지막 열이 라벨
    return data[:, :-1], data[:, -1]


def load_data():
    train = _read_arff(file_path + train_fn)
    test = _read_arff(file_path + test_fn)
    return _split_xy(train), _split_xy(test)
