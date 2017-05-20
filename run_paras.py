from functools import reduce

from extract_features import *
from scipy.stats import *
from sklearn import tree, ensemble, svm, linear_model, neural_network
from concurrent.futures import *
import time


def get_bins(num_bins, flatten_arr):
    assert 0 < num_bins < 100 and isinstance(num_bins, int) and 100 % num_bins is 0
    arr_bins = []
    for per in range(0, 101, int(100 / num_bins)):
        arr_bins.append(np.asscalar(np.percentile(flatten_arr, per)))
    return arr_bins


def average_predict(train_result, test_list) -> np.ndarray:
    num_days = np.size(test_list, 0)
    return np.tile(np.expand_dims(np.mean(train_result, axis=0), 0), (num_days, 1))


def knn_predict(train_list, train_result, test_list, k=9) -> np.ndarray:
    assert np.size(train_list, 1) == np.size(test_list, 1)
    assert np.size(train_list, 0) == np.size(train_result, 0)
    train_list = np.asarray(train_list, dtype=np.float64)
    test_list = np.asarray(test_list, dtype=np.float64)
    squared_diff = (np.expand_dims(test_list, 1) - np.expand_dims(train_list, 0)) ** 2  # n_test * n_train
    distances = np.sum(squared_diff, axis=2)
    k_neighbors = np.argsort(distances, axis=1)[:, :k]  # n_test * k

    result = np.mean(train_result[k_neighbors], axis=1).astype(int)  # n_test * d_result
    return result


def prepare_data(num_bins, **kwargs):
    v_train, _, _ = extract_volume_knn("./training/volume(table 6)_training.csv", "volume_for_knn_train.npy", **kwargs)
    v_test, all_ids, all_days = extract_volume_knn("./testing_phase1/volume(table 6)_test1.csv",
                                                   "volume_for_knn_test.npy", **kwargs)
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


def predict(_train_list, _train_result, _test_list, __method_list, **kwargs):
    def fit_predict_each_output(__model):
        __predict_result = []
        for idx in range(np.size(_train_result, 1)):
            __model.fit(_train_list, _train_result[:, idx])
            __predict_result.append(__model.predict(_test_list))
        return np.transpose(np.asarray(__predict_result))

    def fit_predict(__model):
        __model.fit(_train_list, _train_result)
        return __model.predict(_test_list)

    from_bins_idx = kwargs["from_bins_idx"]
    _predict_result = []
    if "knn" in __method_list:
        _predict_result.append(
            knn_predict(_train_list, _train_result, _test_list, k=kwargs["k"]))
    elif "dt" in __method_list:
        _predict_result.append(fit_predict(tree.DecisionTreeClassifier(max_depth=kwargs["max_depth"])))
    elif "rf" in __method_list:
        _predict_result.append(fit_predict(ensemble.RandomForestClassifier(n_estimators=kwargs["n_estimators"], max_depth=kwargs["max_depth"], n_jobs=kwargs["n_jobs"])))
    elif "average" in __method_list:
        _predict_result.append(average_predict(_train_result, _test_list))
    elif "adaboost" in __method_list:
        _predict_result.append(fit_predict_each_output(ensemble.AdaBoostClassifier()))
    elif "ridge" in __method_list:
        _predict_result.append(fit_predict_each_output(linear_model.RidgeClassifier()))

    elif "linear" in __method_list:
        _predict_result.append(fit_predict_each_output(linear_model.LinearRegression()))
        # return np.asarray(_predict_result)
    elif "huber" in __method_list:
        _predict_result.append(fit_predict_each_output(linear_model.HuberRegressor()))
        # return np.asarray(_predict_result)
    elif "theilsen" in __method_list:
        _predict_result.append(fit_predict_each_output(linear_model.TheilSenRegressor()))
        # return np.asarray(_predict_result)
    elif "lasso" in __method_list:
        _predict_result.append(fit_predict_each_output(linear_model.Lasso()))
        # return np.asarray(_predict_result)
    elif "par" in __method_list:
        _predict_result.append(fit_predict_each_output(linear_model.PassiveAggressiveRegressor()))
        # return np.asarray(_predict_result)
    elif "ridge_reg" in __method_list:
        _predict_result.append(fit_predict_each_output(linear_model.Ridge()))
        # return np.asarray(_predict_result)
    else:
        assert False, "invalid method"
    return np.asarray(_predict_result)
    # return from_bins_idx(np.asarray(_predict_result, dtype=int))


def run(method_list, inputs, cross_validate=True, fold=5, **kwargs):
    def sum_tws(data, num_sum):
        return np.transpose(np.sum(np.split(data, int(np.size(data, 1) / num_sum), axis=1), axis=2))
    v_train = inputs["v_train"]
    v_test = inputs["v_test"]
    all_ids = inputs["all_ids"]
    all_days = inputs["all_days"]
    from_bins_idx = inputs["from_bins_idx"]
    to_bins_idx = inputs["to_bins_idx"]
    INPUT_TW = inputs["input_tw"]
    REQUIRED_TW = inputs["required_tw"]

    if cross_validate:  # cross validate
        rst_sum, rst_count = 0, 0
        train_day_idx_partition = int(np.size(v_train, 2) / fold)
        day_idx = np.arange(0, np.size(v_train, 2))
        np.random.shuffle(day_idx)
        day_idx_test, day_idx_train = day_idx[:train_day_idx_partition], day_idx[train_day_idx_partition:]
        for tollgate_id in range(np.size(v_train, 0)):
            for direction in range(np.size(v_train, 1)):
                i_tw = reduce(lambda x, y: x + y, INPUT_TW)
                r_tw = reduce(lambda x, y: x + y, REQUIRED_TW)
                _train = v_train[tollgate_id, direction, day_idx_train, :, :]
                _test = v_train[tollgate_id, direction, day_idx_test, :, :]
                # for Classifier
                # _train_list = np.concatenate([to_bins_idx(_train[:, i_tw, 0]), _train[:, 0, 1:]], axis=1)
                # _train_result = to_bins_idx(_train[:, r_tw, 0])
                # _test_list = np.concatenate([to_bins_idx(_test[:, i_tw, 0]), _test[:, 0, 1:]], axis=1)

                # for Regressor
                _train_list = np.concatenate([_train[:, i_tw, 0], _train[:, 0, 1:]], axis=1)
                _train_result = _train[:, r_tw, 0]
                _test_list = np.concatenate([_test[:, i_tw, 0], _test[:, 0, 1:]], axis=1)

                _test_result = _test[:, r_tw, 0]
                _test_result = sum_tws(_test_result, inputs["num_sum"])

                _predict_result = predict(_train_list, _train_result, _test_list, method_list, **kwargs, from_bins_idx=from_bins_idx)
                _predict_result = np.mean(_predict_result, axis=0)
                _predict_result = sum_tws(_predict_result, inputs["num_sum"])

                rst_count += np.size(_predict_result)
                with np.errstate(divide='ignore', invalid='ignore'):
                    diff = np.abs(_predict_result[_test_result != 0] - _test_result[_test_result != 0])
                    fraction = diff / (_test_result[_test_result != 0] + np.asarray((EPS,)))
                    fraction[diff == 0] = 0
                    rst_sum += np.sum(fraction)
        mape = rst_sum / rst_count
        return mape
    else:  # test
        tw_seconds = inputs["time_window_seconds"]
        test_output_file = open("volume_output_%s_%d.csv" % ("_".join(method_list), int(datetime.now().timestamp())), "w+")
        print("tollgate_id, time_window, direction, volume", file=test_output_file)
        for tollgate_id in range(np.size(v_train, 0)):
            for direction in range(np.size(v_train, 1)):
                i_tw = reduce(lambda x, y: x + y, INPUT_TW)
                r_tw = reduce(lambda x, y: x + y, REQUIRED_TW)
                _train = v_train[tollgate_id, direction, :, :, :]
                _test = v_test[tollgate_id, direction, :, :, :]
                # _train_list = np.concatenate([to_bins_idx(_train[:, i_tw, 0]), _train[:, 0, 1:]], axis=1)
                # _train_result = to_bins_idx(_train[:, r_tw, 0])
                # _test_list = np.concatenate([to_bins_idx(_test[:, i_tw, 0]), _test[:, 0, 1:]], axis=1)

                _train_list = np.concatenate([_train[:, i_tw, 0], _train[:, 0, 1:]], axis=1)
                _train_result = _train[:, r_tw, 0]
                _test_list = np.concatenate([_test[:, i_tw, 0], _test[:, 0, 1:]], axis=1)

                _predict_result = predict(_train_list, _train_result, _test_list, method_list, **kwargs, from_bins_idx=from_bins_idx)
                _predict_result = np.mean(_predict_result, axis=0)
                _predict_result = sum_tws(_predict_result, inputs["num_sum"])

                for day in range(np.size(v_test, 2)):
                    for idx in range(0, len(r_tw), inputs["num_sum"]):
                        tw = r_tw[idx]
                        timestamp = int(all_days[day]) * 86400 - 3600 * 8 + tw_seconds * tw
                        print("%s,\"[%s,%s)\",%s,%f" % (
                            all_ids[tollgate_id], datetime.fromtimestamp(timestamp).isoformat(" "),
                            datetime.fromtimestamp(timestamp + 1200).isoformat(" "), direction,
                            np.asscalar(_predict_result[day, int(idx / inputs["num_sum"])])), file=test_output_file)
        test_output_file.close()


def test_method(rounds, method_list, inputs, **kwargs):
    mape_list = []
    for i in range(rounds):
        mape_list.append(run(method_list, inputs, **kwargs))
        # if i % 5 == 0:
        print("finished rounds: ", i)
        time.sleep(15)
    print("Average MAPE of %s = " % str(method_list), np.mean(mape_list))


def main():
    rounds = 100
    num_bin = 51
    params = {
        "k": 7,
        "max_depth": 4,
        "n_estimators": 100,
        "n_jobs": 8
    }
    tw_seconds = 60

    v_train, v_test, all_ids, all_days, volume_bins = prepare_data(num_bin, time_window_seconds=tw_seconds, refresh=False)

    def from_bins_idx(arr):
        return np.vectorize(volume_bins.__getitem__)(arr)

    def to_bins_idx(arr):
        return np.searchsorted(volume_bins, arr)

    inputs = {
        "v_train": v_train,
        "v_test": v_test,
        "from_bins_idx": from_bins_idx,
        "to_bins_idx": to_bins_idx,
        "all_ids": all_ids,
        "all_days": all_days,
        "input_tw": (tuple(range(int(6 * 3600 / tw_seconds), int(8 * 3600 / tw_seconds))), tuple(range(int(15 * 3600 / tw_seconds), int(17 * 3600 / tw_seconds)))),
        "required_tw": (tuple(range(int(8 * 3600 / tw_seconds), int(10 * 3600 / tw_seconds))), tuple(range(int(17 * 3600 / tw_seconds), int(19 * 3600 / tw_seconds)))),
        "num_sum": int(1200 / tw_seconds),
        "time_window_seconds": tw_seconds,
    }

    # test_method(rounds, ["adaboost"], inputs, **params)
    # test_method(rounds, ["ridge"], inputs, **params)
    # test_method(rounds, ["dt"], inputs, **params)
    # test_method(rounds, ["average"], inputs, **params)
    # test_method(rounds, ["knn"], inputs, **params)
    # test_method(rounds, ["knn", "ridge"], inputs, **params)
    # test_method(rounds, ["dt", "ridge"], inputs, **params)
    # test_method(rounds, ["dt", "ridge", "knn"], inputs, **params)
    # test_method(rounds, ["rf"], inputs, **params)
    # run(["ridge"], inputs, False, **params)

    # test_method(rounds, ["linear"], inputs, **params)
    # test_method(rounds, ["huber"], inputs, **params)
    # test_method(rounds, ["theilsen"], inputs, **params)
    # test_method(rounds, ["par"], inputs, **params)
    # test_method(rounds, ["ridge_reg"], inputs, **params)
    run(['theilsen', 'huber'], inputs, False, **params)

    # test_method(rounds, ["huber", "ridge_reg"], inputs, **params)
    # test_method(rounds, ["huber", "theilsen"], inputs, **params)


if __name__ == '__main__':
    thread_pool = ThreadPoolExecutor(max_workers=4)
    EPS = 1e-8
    main()
