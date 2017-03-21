# Some utility functions
def log(*args):
    from datetime import datetime
    now = datetime.now()
    display_now = str(now).split(" ")[1][:-3]
    print(display_now, *args)


def last_timewindow(timestamp, timewindow=20*60) -> int:
    return timestamp - timewindow


def timestamp2daily(timestamp):
    import numpy as np
    return np.remainder(timestamp, 86400)


def fold_data(features, label, fold):
    import numpy as np
    assert np.size(features, axis=0) == np.size(label, axis=0)
    length = np.size(label, axis=0)
    index = np.arange(length)
    np.random.shuffle(index)
    features = features[index]
    label = label[index]
    partition = int(length / fold)
    return features[partition:], label[partition:], features[:partition], label[:partition]


def accuracy(x, y):
    import numpy as np
    assert np.size(x) == np.size(y)
    return 1 - np.count_nonzero(x - y) / np.size(x)

