# the naive model
# Do keep everything local as possible as you can.
from typing import Tuple

from pandas import DataFrame
from sklearn import preprocessing, tree, model_selection, ensemble, linear_model, svm, neural_network
from extract_features import *
import pydotplus
import argparse
import seaborn as sns
from matplotlib import pyplot as plt


def main():
    if FLAGS.clean:
        remove_is_exist(FLAGS.weather_raw_data)
        remove_is_exist(FLAGS.volume_raw_data)
        remove_is_exist(FLAGS.travel_time_raw_data)
        remove_is_exist(FLAGS.volume_feature)
        remove_is_exist(FLAGS.travel_time_feature)
    if not os.path.exists(FLAGS.volume_feature) or not os.path.exists(FLAGS.travel_time_feature):
        v, tt = prepare_data("naive")
        np.savetxt(FLAGS.volume_feature, v)
        np.savetxt(FLAGS.travel_time_feature, tt)
    else:
        v = np.loadtxt(FLAGS.volume_feature)
        tt = np.loadtxt(FLAGS.travel_time_feature)
    log("data preparation finished")
    log("volume shape:", np.shape(v))
    log("travel time shape:", np.shape(tt))

    test_some_models(v, tt)
    # analyse_features_by_plotting(v, tt)


def analyse_features_by_plotting(v, tt):
    v_frame = DataFrame(v, columns=["tollgate id", "direction", "last volume", "pressure", "sea pressure", "wind direction", "wind speed", "temperature", "rel_humidity", "precipitation", "daily time", "volume"])
    volume = v[:, -1]
    daily_time = v[:, 10]
    plot_x_y(daily_time, volume)
    plt.show()


def test_some_models(v, tt):
    v_feature_train, v_label_train, v_feature_test, v_label_test = fold_data(v[:, :-1], v[:, -1], FLAGS.fold)
    t_feature_train, t_label_train, t_feature_test, t_label_test = fold_data(tt[:, :-1], tt[:, -1], FLAGS.fold)

    volume_percentiles = np.asarray([np.percentile(v_label_train, i * 10) for i in range(10)])
    v_label_train = np.searchsorted(volume_percentiles, v_label_train)
    v_label_test = np.searchsorted(volume_percentiles, v_label_test)

    travel_time_percentiles = np.asarray([np.percentile(t_label_train, i * 10) for i in range(10)])
    t_label_train = np.searchsorted(travel_time_percentiles, t_label_train)
    t_label_test = np.searchsorted(travel_time_percentiles, t_label_test)

    def use_a_model(model, name, f_train, l_train, f_test, l_test):
        model.fit(f_train, l_train)
        _acc = accuracy(model.predict(f_test), l_test)
        log("%s accuracy with %s: %f" % (name, type(model).__name__, _acc))
        return model

    def model_v(model): return use_a_model(model, "volume", v_feature_train, v_label_train, v_feature_test, v_label_test)

    def model_t(model): return use_a_model(model, "travel time", t_feature_train, t_label_train, t_feature_test, t_label_test)

    # model_v(tree.DecisionTreeClassifier())
    # model_t(tree.DecisionTreeClassifier())
    # # require graphviz installed. use `sudo apt install graphviz` on Ubuntu or comment codes below
    # # dot_data = tree.export_graphviz(t_dt, out_file=None, max_depth=4)
    # # graph = pydotplus.graph_from_dot_data(dot_data)
    # # graph.write_pdf("travel_time_decision_tree.pdf")
    #
    # model_v(ensemble.RandomForestClassifier(n_estimators=1000))
    # model_t(ensemble.RandomForestClassifier(n_estimators=1000))
    #
    # model_v(svm.SVC())
    # model_t(svm.SVC())

    model_v(neural_network.MLPClassifier(max_iter=100000))
    model_t(neural_network.MLPClassifier(max_iter=100000))


def prepare_data(method) -> Tuple[np.ndarray, np.ndarray]:
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

parser.add_argument("--clean", action="store_true")

FLAGS = parser.parse_args()
if __name__ == "__main__":
    main()
