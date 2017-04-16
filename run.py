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
        v.to_csv(FLAGS.volume_feature, index=False)
        tt.to_csv(FLAGS.travel_time_feature, index=False)
    else:
        v = DataFrame.from_csv(FLAGS.volume_feature, index_col=None)
        tt = DataFrame.from_csv(FLAGS.travel_time_feature, index_col=None)
    log("data preparation finished")
    log("volume shape:", v.shape)
    log("travel time shape:", tt.shape)

    # test_some_models(v, tt)
    analyse_features_by_plotting(v, tt)


def analyse_features_by_plotting(v, tt):
    clean_dir("./pic")
    sns.boxplot(x="lav", y="cav", data=v)
    sns.plt.savefig("./pic/last_volume_to_volume.jpg")
    sns.boxplot(x="time", y="cav", data=v)
    sns.plt.savefig("./pic/time_to_volume.jpg")
    sns.boxplot(x="day", y="cav", data=v)
    sns.plt.savefig("./pic/day_to_volume.jpg")
    sns.boxplot(x="tid", y="cav", data=v)
    sns.plt.savefig("./pic/tid_to_volume.jpg")
    sns.boxplot(x="dir", y="cav", data=v)
    sns.plt.savefig("./pic/dir_to_volume.jpg")
    ###
    sns.jointplot(x="lav", y="cat", data=tt)
    sns.plt.savefig("./pic/last_volume_to_average_travel_time.jpg")
    sns.boxplot(x="time", y="cat", data=tt)
    sns.plt.savefig("./pic/time_to_average_travel_time.jpg")
    sns.boxplot(x="day", y="cat", data=tt)
    sns.plt.savefig("./pic/day_to_average_travel_time.jpg")
    sns.jointplot(x="lat", y="cat", data=tt)
    sns.plt.savefig("./pic/last_travel_time_to_average_travel_time.jpg")
    sns.boxplot(x="tid", y="cat", data=tt)
    sns.plt.savefig("./pic/tid_to_average_travel_time.jpg")
    sns.boxplot(x="iid", y="cat", data=tt)
    sns.plt.savefig("./pic/iid_to_average_travel_time.jpg")


def test_some_models(v, tt):
    _v_tmp = fold_data(np.asarray(v[["tid", "lav", "time", "day"]]), np.asarray(v["cav"]), FLAGS.fold)
    _tt_tmp = fold_data(np.asarray(tt[["tid", "iid", "lav", "lat", "time", "day"]]), np.asarray(tt["cat"]), FLAGS.fold)

    v_feature_train, v_label_train, v_feature_test, v_label_test = _v_tmp
    t_feature_train, t_label_train, t_feature_test, t_label_test = _tt_tmp

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

    v_dt = model_v(tree.DecisionTreeClassifier())
    t_dt = model_t(tree.DecisionTreeClassifier())
    # require graphviz installed. use `sudo apt install graphviz` on Ubuntu or comment codes below
    dot_data = tree.export_graphviz(v_dt, out_file=None, max_depth=4, feature_names=["tid", "lav", "time", "day"])
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("travel_time_decision_tree.pdf")

    model_v(ensemble.RandomForestClassifier(n_estimators=1000))
    model_t(ensemble.RandomForestClassifier(n_estimators=1000))

    model_v(svm.SVC())
    model_t(svm.SVC())

    model_v(neural_network.MLPClassifier(max_iter=1000000))
    model_t(neural_network.MLPClassifier(max_iter=1000000))


def prepare_data(method) -> Tuple[DataFrame, DataFrame]:
    if method is "naive":
        return prepare_data_naive()


def prepare_data_naive():
    # Note that the sort is required
    # Note that the index of features is hard coded, be careful
    weather_data = extract_weather(FLAGS.weather_input, FLAGS.weather_raw_data)
    weather_data = weather_data[weather_data[:, 0].argsort()]
    volume_data = extract_volume_naive(FLAGS.volume_input, FLAGS.volume_raw_data, weather_data)
    volume_data = volume_data[volume_data[:, -1].argsort()]
    travel_time_data = extract_travel_time_naive(FLAGS.travel_time_input, FLAGS.travel_time_raw_data, volume_data)
    # get really useful features from somewhat raw data
    volume = volume_data[:, [0, 1, 2, 3, 11, 11]]
    travel_time = travel_time_data[:, [0, 1, 2, 3, 4, 12, 12]]
    # convert timestamp to time in a day
    volume[:, -2] = timestamp2daily(volume[:, -2])
    travel_time[:, -2] = timestamp2daily(travel_time[:, -2])
    volume[:, -1] = timestamp2day_of_week(volume[:, -1])
    travel_time[:, -1] = timestamp2day_of_week(travel_time[:, -1])
    # normalize data
    volume = np.concatenate([volume[:, :2], zero_normalization(volume[:, 2:-1]), volume[:, -1:]], 1)
    travel_time = np.concatenate([travel_time[:, :2], zero_normalization(travel_time[:, 2:-1]), travel_time[:, -1:]], 1)
    # convert to DataFrame
    volume = DataFrame(data=volume, columns=["tid", "dir", "lav", "cav", "time", "day"])
    travel_time = DataFrame(data=travel_time, columns=["tid", "iid", "lat", "cat", "lav", "time", "day"])
    print(volume)
    print(travel_time)
    return volume, travel_time


parser = argparse.ArgumentParser(description="KDD CUP 2017. The answer of universe")

parser.add_argument("--volume_feature", default="volume_preprocessed.csv", type=str)
parser.add_argument("--travel_time_feature", default="travel_time_preprocessed.csv", type=str)

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
