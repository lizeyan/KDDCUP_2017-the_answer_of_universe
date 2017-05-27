# Some utility functions
import shutil

from matplotlib import pyplot as plt
import numpy as np
import os

holiday_set = {17059, 17060, 17061, 17075, 17076, 17077, 17078, 17079, 17080, 17081}


def log(*args):
    from datetime import datetime
    now = datetime.now()
    display_now = str(now).split(" ")[1][:-3]
    print(display_now, *args)


def last_timewindow(timestamp, timewindow=20 * 60) -> int:
    return timestamp - timewindow


def timestamp2daily(timestamp):
    return np.remainder(np.add(timestamp, 3600 * 8), 86400)


def timestamp2day_of_week(timestamp):
    return np.remainder((np.divide(np.remainder(np.add(timestamp, 3600 * 8), 86400 * 7), 86400) + 4), 7).astype(int)


def timestamp2day(timestamp):
    return np.divide(timestamp + 3600 * 8, 86400).astype(int)


def fold_data(features, label, fold):
    assert np.size(features, axis=0) == np.size(label, axis=0)
    length = np.size(label, axis=0)
    index = np.arange(length)
    np.random.shuffle(index)
    features = features[index]
    label = label[index]
    partition = int(length / fold)
    return features[partition:], label[partition:], features[:partition], label[:partition]


def accuracy(x, y):
    assert np.size(x) == np.size(y)
    return 1 - np.count_nonzero(x - y) / np.size(x)


def plot_x_y(x, y, bins=10):
    length = 100 / bins
    x_bins = np.asarray([np.percentile(x, length * i) for i in range(bins)])
    x = np.searchsorted(x_bins, x)
    x_bins = np.append(x_bins, np.max(x))
    x_plot = []
    y_plot = []
    for i in range(bins):
        x_plot.append(np.mean(x_bins[[i, i + 1]]))
        y_plot.append(np.mean(y[x == i]))

    plt.plot(x_plot, y_plot)


def remove_is_exist(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def zero_normalization(array):
    """
    normalize each column of array to zero mean and unit variance
    """
    return (array - np.mean(array, axis=0)) / np.std(array, axis=0)


def is_holiday(timestamp):
    return np.vectorize(lambda x: 1 if int((x + 3600 * 8) / 86400) in holiday_set else 0)(timestamp)


def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def rain_level(precipitation):
    if precipitation == 0:
        return 0
    elif precipitation < 990:
        return 1
    elif precipitation < 2490:
        return 2
    elif precipitation < 4990:
        return 3
    else:
        return 4


def rstrip_str(s: str, rend: str):
    if s.endswith(rend):
        return s[:-len(rend)]
    else:
        return s
