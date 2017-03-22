# Some utility functions
from matplotlib import pyplot as plt
import numpy as np
import os


def log(*args):
    from datetime import datetime
    now = datetime.now()
    display_now = str(now).split(" ")[1][:-3]
    print(display_now, *args)


def last_timewindow(timestamp, timewindow=20*60) -> int:
    return timestamp - timewindow


def timestamp2daily(timestamp):
    return np.remainder(timestamp, 86400)


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
