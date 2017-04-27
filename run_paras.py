from datetime import *
import numpy as np
from extract_features import *
from scipy.stats import *
from sklearn import tree, ensemble
from concurrent.futures import *


def get_bins(num_bins, flatten_arr):
    assert 0 < num_bins < 100 and isinstance(num_bins, int) and 100 % num_bins is 0
    arr_bins = []
    for per in range(0, 101, int(100 / num_bins)):
        arr_bins.append(np.asscalar(np.percentile(flatten_arr, per)))
    return arr_bins


def knn_predict(train_list, train_result, test_list, k=9, feature_weights=None) -> np.ndarray:
    assert np.size(train_list, 1) == np.size(test_list, 1)
    assert np.size(train_list, 0) == np.size(train_result, 0)
    d_feature = np.size(train_list, 1)
    # d_result = np.size(train_result, 1)
    if feature_weights is None:
        feature_weights = np.ones(d_feature)
    train_list = np.asarray(train_list, dtype=np.float64)
    test_list = np.asarray(test_list, dtype=np.float64)
    squared_diff = (np.expand_dims(test_list, 1) - np.expand_dims(train_list, 0)) ** 2  # n_test * n_train
    squared_diff *= np.expand_dims(np.expand_dims(feature_weights, 0), 0)
    distances = np.sum(squared_diff, axis=2)
    k_neighbors = np.argsort(distances, axis=1)[:, :k]  # n_test * k
    result = mode(train_result[k_neighbors], axis=1)[0]  # n_test * d_result
    return np.squeeze(result, 1)


def prepare_data(num_bins):
    v_train, _, _ = extract_volume_knn("./training/volume(table 6)_training.csv", "volume_for_knn_train.npy")
    v_test, all_ids, all_days = extract_volume_knn("./testing_phase1/volume(table 6)_test1.csv",
                                                   "volume_for_knn_test.npy")
    # log("Train data shape:", np.shape(v_train))
    # log("Test data shape:", np.shape(v_test))

    AVERAGE_VOLUME = 0
    DAY_OF_WEEK = 1
    all_volume = np.concatenate(
        [v_train[:, :, :, :, AVERAGE_VOLUME].flatten(), v_test[:, :, :, :, AVERAGE_VOLUME].flatten()])
    volume_bins = get_bins(num_bins - 1, all_volume[all_volume > 0])
    volume_bins = [0] + volume_bins
    volume_predict = np.asarray(volume_bins.copy(), dtype=float)
    volume_predict[:-1] = (volume_predict[:-1] + volume_predict[1:]) / 2

    return v_train, v_test, all_ids, all_days, volume_bins


def run(method_list, v_train, v_test, all_ids, all_days, from_bins_idx, to_bins_idx, cross_validate=True, fold=10, **kwargs):
    INPUT_TW = ((18, 19, 20, 21, 22, 23), (45, 46, 47, 48, 49, 50))
    REQUIRED_TW = ((24, 25, 26, 27, 28, 29), (51, 52, 53, 54, 55, 56))

    # all params:
    feature_weight, k, n_estimators, max_depth = None, None, None, None
    if "knn" in method_list:
        feature_weight = [0.1, 0.2, 0.4, 0.7, 0.9, 1.0, kwargs["weight1"], kwargs["weight2"]]
        k = kwargs["k"]
    elif "rf" in method_list:
        n_estimators = kwargs["n_estimators"]
        max_depth = kwargs["max_depth"]
    elif "dt" in method_list:
        max_depth = kwargs["max_depth"]
    else:
        assert False, "invalid method"

    if cross_validate:  # cross validate
        rst_sum, rst_count = 0, 0
        train_day_idx_partition = int(np.size(v_train, 2) / fold)
        day_idx = np.arange(0, np.size(v_train, 2))
        np.random.shuffle(day_idx)
        day_idx_test, day_idx_train = day_idx[:train_day_idx_partition], day_idx[train_day_idx_partition:]
        for tollgate_id in range(np.size(v_train, 0)):
            for direction in range(np.size(v_train, 1)):
                for i_tw, r_tw in zip(INPUT_TW, REQUIRED_TW):
                    _train = v_train[tollgate_id, direction, day_idx_train, :, :]
                    _test = v_train[tollgate_id, direction, day_idx_test, :, :]
                    _train_result = to_bins_idx(_train[:, r_tw, 0])
                    _train_list = np.concatenate([to_bins_idx(_train[:, i_tw, 0]), _train[:, 0, 1:]], axis=1)
                    _test_list = np.concatenate([to_bins_idx(_test[:, i_tw, 0]), _test[:, 0, 1:]], axis=1)
                    _test_result = _test[:, r_tw, 0]

                    _predict_result = []
                    # Choose a model (by command argv[5])
                    if "knn" in method_list:  # KNN
                        _predict_result.append(from_bins_idx(
                            knn_predict(_train_list, _train_result, _test_list, k=k, feature_weights=feature_weight)))
                    elif "dt" in method_list:  # Decision Tree
                        model = tree.DecisionTreeClassifier(max_depth=max_depth)
                        model.fit(_train_list, _train_result)
                        _predict_result.append(from_bins_idx(model.predict(_test_list).astype(int)))
                    elif "rf" in method_list:  # Random Forest
                        model = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                        model.fit(_train_list, _train_result)
                        _predict_result.append(from_bins_idx(model.predict(_test_list).astype(int)))
                    else:
                        assert False, "invalid method"
                    _predict_result = np.mean(_predict_result, axis=0)
                    rst_count += len(r_tw)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        diff = np.abs(_predict_result[_test_result != 0] - _test_result[_test_result != 0])
                        fraction = diff / (_test_result[_test_result != 0] + np.asarray((EPS,)))
                        fraction[diff == 0] = 0
                        rst_sum += np.sum(fraction)
        mape = rst_sum / rst_count * 2
        # log("Cross validate MAPE by %s" % method, mape)
        return mape
    else:  # test
        test_output_file = open("volume_output.csv", "w+")
        print("tollgate_id, time_window, direction, volume", file=test_output_file)
        for tollgate_id in range(np.size(v_train, 0)):
            for direction in range(np.size(v_train, 1)):
                for i_tw, r_tw in zip(INPUT_TW, REQUIRED_TW):
                    _train = v_train[tollgate_id, direction, :, :, :]
                    _test = v_test[tollgate_id, direction, :, :, :]
                    _train_list = np.concatenate([to_bins_idx(_train[:, i_tw, 0]), _train[:, 0, 1:]], axis=1)
                    _train_result = to_bins_idx(_train[:, r_tw, 0])
                    _test_list = np.concatenate([to_bins_idx(_test[:, i_tw, 0]), _test[:, 0, 1:]], axis=1)
                    _predict_result = []
                    if "knn" in method_list:  # KNN
                        _predict_result.append(from_bins_idx(
                            knn_predict(_train_list, _train_result, _test_list, k=k, feature_weights=feature_weight)))
                    elif "dt" in method_list:  # Decision Tree
                        model = tree.DecisionTreeClassifier(max_depth=max_depth)
                        model.fit(_train_list, _train_result)
                        _predict_result.append(from_bins_idx(model.predict(_test_list).astype(int)))
                    elif "rf" in method_list:  # Random Forest
                        model = ensemble.RandomForestClassifier(n_estimators=n_estimators)
                        model.fit(_train_list, _train_result)
                        _predict_result.append(from_bins_idx(model.predict(_test_list).astype(int)))
                    else:
                        assert False, "invalid method"
                    _predict_result = np.mean(_predict_result, axis=0)
                    for day in range(np.size(v_test, 2)):
                        for idx, tw in enumerate(r_tw):
                            timestamp = int(all_days[day]) * 86400 - 3600 * 8 + 1200 * tw
                            print("%s,\"[%s,%s)\",%s,%f" % (
                                all_ids[tollgate_id], datetime.fromtimestamp(timestamp).isoformat(" "),
                                datetime.fromtimestamp(timestamp + 1200).isoformat(" "), direction,
                                np.asscalar(_predict_result[day, idx])), file=test_output_file)
        test_output_file.close()


def test_method(rounds, method_list, v_train, v_test, all_ids, all_days, from_bins_idx, to_bins_idx, **kwargs):

    mape_list = []
    for i in range(rounds):
        mape_list.append(
            run(method_list, v_train=v_train, v_test=v_test, all_ids=all_ids, all_days=all_days,
                from_bins_idx=from_bins_idx, to_bins_idx=to_bins_idx, **kwargs))
    # thread_pool.shutdown(wait=True)
    print("Average MAPE of %s = " % str(method_list), np.mean(mape_list))


def main():
    rounds = 1000
    num_bin = 51
    weight2 = 0.9
    weight1 = 0.7
    k = 6
    max_depth = 3
    n_estimator = 100
    v_train, v_test, all_ids, all_days, volume_bins = prepare_data(num_bin)

    def from_bins_idx(arr):
        return np.vectorize(volume_bins.__getitem__)(arr)

    def to_bins_idx(arr):
        return np.searchsorted(volume_bins, arr)

    test_method(rounds, "dt", v_train, v_test, all_ids, all_days, from_bins_idx, to_bins_idx, max_depth=max_depth)
    test_method(rounds, ["dt", "knn", "rf"], v_train, v_test, all_ids, all_days, from_bins_idx, to_bins_idx, max_depth=max_depth, weight1=weight1, weight2=weight2, k=k, n_estimator=n_estimator)
    # test_method(rounds, "rf", v_train, v_test, all_ids, all_days, from_bins_idx, to_bins_idx, n_estimators=5)
    run(["dt"], v_train, v_test, all_ids, all_days, from_bins_idx, to_bins_idx, False, max_depth=max_depth)
    run(["dt", "knn", "rf"], v_train, v_test, all_ids, all_days, from_bins_idx, to_bins_idx, False, max_depth=max_depth, weight1=weight1, weight2=weight2, k=k, n_estimator=n_estimator)


if __name__ == '__main__':
    thread_pool = ThreadPoolExecutor(max_workers=4)
    EPS = 1e-8
    main()
