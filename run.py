# the naive model
# Do keep everything local as possible as you can.
from typing import Tuple
from sklearn import preprocessing, tree
from extract_features import *
import argparse


def main():
    if not os.path.exists(FLAGS.volume_feature) or not os.path.exists(FLAGS.travel_time_feature):
        v, tt = prepare_data("naive")
        np.savetxt(FLAGS.volume_feature, v)
        np.savetxt(FLAGS.travel_time_feature, tt)
    else:
        v = np.loadtxt(FLAGS.volume_feature)
        tt = np.loadtxt(FLAGS.travel_time_feature)
    log("volume shape:", np.shape(v))
    log("travel time shape:", np.shape(tt))
    v_feature_train, v_label_train, v_feature_test, v_label_test = fold_data(v[:, :-1], v[:, -1], FLAGS.fold)
    t_feature_train, t_label_train, t_feature_test, t_label_test = fold_data(tt[:, :-1], tt[:, -1], FLAGS.fold)

    volume_percentiles = np.asarray([np.percentile(v_label_train, i * 10) for i in range(10)])
    v_label_train = np.searchsorted(volume_percentiles, v_label_train)
    v_label_test = np.searchsorted(volume_percentiles, v_label_test)

    travel_time_percentiles = np.asarray([np.percentile(t_label_train, i * 10) for i in range(10)])
    t_label_train = np.searchsorted(travel_time_percentiles, t_label_train)
    t_label_test = np.searchsorted(travel_time_percentiles, t_label_test)

    volume_clf = tree.DecisionTreeClassifier()
    volume_clf.fit(v_feature_train, v_label_train)
    v_accuracy = accuracy(volume_clf.predict(v_feature_test), v_label_test)
    log("volume accuracy: %f" % v_accuracy)

    travel_time_clf = tree.DecisionTreeClassifier()
    travel_time_clf.fit(t_feature_train, t_label_train)
    t_accuracy = accuracy(travel_time_clf.predict(t_feature_test), t_label_test)
    log("travel time accuracy: %f" % t_accuracy)


def prepare_data(method, clean=False) -> Tuple[np.ndarray, np.ndarray]:
    if clean:
        os.remove(FLAGS.weather_raw_data)
        os.remove(FLAGS.volume_raw_data)
        os.remove(FLAGS.travel_time_raw_data)

    weather_data, volume_data, travel_time_data = None, None, None
    if method is "naive":
        # Note that the sort is required
        # Note that the index of features is hard coded, be careful
        weather_data = extract_weather(FLAGS.weather_input, FLAGS.weather_raw_data)
        weather_data = weather_data[weather_data[:, 0].argsort()]
        volume_data = extract_volume_naive(FLAGS.volume_input, FLAGS.volume_raw_data, weather_data)
        volume_data = volume_data[volume_data[:, -1].argsort()]
        travel_time_data = extract_travel_time_naive(FLAGS.travel_time_input, FLAGS.travel_time_raw_data, volume_data)
    # get really useful features from somewhat raw data
    volume = volume_data[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 3]]
    travel_time = travel_time_data[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3]]
    # convert timestamp to time in a day
    volume[:, -2] = timestamp2daily(volume[:, -2])
    travel_time[:, -2] = timestamp2daily(travel_time[:, -2])
    # normalize data
    volume = preprocessing.normalize(volume, axis=0, norm="l2")
    travel_time = preprocessing.normalize(travel_time, axis=0, norm="l2")
    return volume, travel_time


parser = argparse.ArgumentParser(description="KDD CUP 2017. The answer of universe")

parser.add_argument("--volume_feature", default="volume_preprocessed.data", type=str)
parser.add_argument("--travel_time_feature", default="travel_time_preprocessed.data", type=str)

parser.add_argument("--volume_raw_data", default="volume.data", type=str)
parser.add_argument("--travel_time_raw_data", default="travel_time.data", type=str)
parser.add_argument("--weather_raw_data", default="weather.data", type=str)

parser.add_argument("--volume_input", default="./training/volume(table 6)_training.csv", type=str)
parser.add_argument("--travel_time_input", default="./training/trajectories(table 5)_training.csv", type=str)
parser.add_argument("--weather_input", default="./training/weather (table 7)_training.csv", type=str)

parser.add_argument("--method", default="naive", choices=["naive"], type=str)
parser.add_argument("--fold", default=10, type=int)
FLAGS = parser.parse_args()
if __name__ == "__main__":
    main()
