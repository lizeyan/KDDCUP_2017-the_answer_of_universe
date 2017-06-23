from functools import reduce
import xgboost as xgb

from extract_features import *
from scipy.stats import *
from sklearn import tree, ensemble, svm, linear_model, neural_network, pipeline
from concurrent.futures import ThreadPoolExecutor


def get_bins_depth(num_bins, flatten_arr):
    assert 0 < num_bins < 100 and isinstance(num_bins, int) and 100 % num_bins is 0
    arr_bins = []
    for per in range(0, 101, int(100 / num_bins)):
        arr_bins.append(np.asscalar(np.percentile(flatten_arr, per)))
    return np.asarray(arr_bins)


def get_bins_width(num_bins, flatten_arr):
    assert 0 < num_bins < 100 and isinstance(num_bins, int)
    minimum = np.min(flatten_arr)
    maximum = np.max(flatten_arr)
    return np.arange(start=minimum, stop=maximum+EPS, step=(maximum - minimum) / num_bins)


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
    v_train, _, _ = extract_volume_knn("./train/volume(table 6)_training.csv", "volume_for_knn_train.npy",
                                       "./train/weather (table 7)_training.csv", **kwargs)

    v_test, all_ids, all_days = extract_volume_knn("./test/volume(table 6)_test2.csv",
                                                   "volume_for_knn_test.npy", "./test/weather (table 7)_2.csv", **kwargs)

    log("Train data shape:", np.shape(v_train))
    log("Test data shape:", np.shape(v_test))

    AVERAGE_VOLUME = 0
    all_volume = np.concatenate(
        [v_train[:, :, :, :, AVERAGE_VOLUME].flatten(), v_test[:, :, :, :, AVERAGE_VOLUME].flatten()])
    # volume_bins = get_bins_depth(num_bins - 1, all_volume[all_volume > 0])
    volume_bins = get_bins_width(num_bins - 1, all_volume[all_volume > 0])
    volume_bins = np.concatenate([[0], volume_bins])

    log("volume bins", volume_bins)

    # v_train[:, :, :, :, AVERAGE_VOLUME] = (v_train[:, :, :, :, AVERAGE_VOLUME] - np.mean(all_volume)) / np.std(all_volume)
    # v_test[:, :, :, :, AVERAGE_VOLUME] = (v_test[:, :, :, :, AVERAGE_VOLUME] - np.mean(all_volume)) / np.std(all_volume)

    return v_train, v_test, all_ids, all_days, volume_bins


def predict(train_list, train_result, test_list, method_list, **kwargs):
    def fit_predict_each_output(model, target):
        __predict_result = []
        for idx in range(np.size(target, 1)):
            model.fit(train_list, target[:, idx])
            __predict_result.append(model.predict(test_list))
        return np.transpose(np.asarray(__predict_result))

    def fit_predict(model, target):
        model.fit(train_list, target)
        return model.predict(test_list)

    from_bins_idx = kwargs["from_bins_idx"]
    to_bins_idx = kwargs["to_bins_idx"]
    _binned_train_result = to_bins_idx(train_result)

    _predict_result = []
    if "current" in method_list:
        rbm = neural_network.BernoulliRBM(n_components=512, verbose=False, n_iter=100, learning_rate=1e-2, random_state=0)
        rbm.fit(train_list)
        rbm.fit(test_list)
        _predict_result.append(np.transpose(np.asarray(__predict_result)))
    elif "knn" in method_list:
        _ = knn_predict(train_list, _binned_train_result, test_list, k=kwargs["k"])
        _predict_result.append(from_bins_idx(np.asarray(_, dtype=int)))
    elif "dt" in method_list:
        _ = fit_predict(tree.DecisionTreeClassifier(max_depth=kwargs["max_depth"]), _binned_train_result)
        _predict_result.append(from_bins_idx(np.asarray(_, dtype=int)))
    elif "rf" in method_list:
        _ = fit_predict(ensemble.RandomForestClassifier(n_estimators=kwargs["n_estimators"], max_depth=kwargs["max_depth"], n_jobs=kwargs["n_jobs"]), _binned_train_result)
        _predict_result.append(from_bins_idx(np.asarray(_, dtype=int)))
    elif "average" in method_list:
        _ = average_predict(train_result, test_list)
        _predict_result.append(from_bins_idx(np.asarray(_, dtype=int)))
    elif "adaboost" in method_list:
        _ = fit_predict_each_output(ensemble.AdaBoostClassifier(), _binned_train_result)
        _predict_result.append(from_bins_idx(np.asarray(_, dtype=int)))
    elif "ridge" in method_list:
        _ = fit_predict_each_output(linear_model.RidgeClassifier(), _binned_train_result)
        _predict_result.append(from_bins_idx(np.asarray(_, dtype=int)))
    elif "linear" in method_list:
        _predict_result.append(fit_predict_each_output(linear_model.LinearRegression(), train_result))
    elif "huber" in method_list:
        _predict_result.append(fit_predict_each_output(linear_model.HuberRegressor(), train_result))
    elif "theilsen" in method_list:
        _predict_result.append(fit_predict_each_output(linear_model.TheilSenRegressor(), train_result))
    elif "lasso" in method_list:
        _predict_result.append(fit_predict_each_output(linear_model.Lasso(), train_result))
    elif "par" in method_list:
        _predict_result.append(fit_predict_each_output(linear_model.PassiveAggressiveRegressor(C=kwargs["par_C"], epsilon=kwargs["par_eps"]), train_result))
    elif "ridge_reg" in method_list:
        _predict_result.append(fit_predict_each_output(linear_model.Ridge(), train_result))
    elif "dt_reg" in method_list:
        _predict_result.append(fit_predict(tree.DecisionTreeRegressor(max_depth=kwargs["max_depth"]), train_result))
    elif "rf_reg" in method_list:
        _predict_result.append(fit_predict(ensemble.RandomForestRegressor(max_depth=kwargs["max_depth"], n_jobs=kwargs['n_jobs'], n_estimators=kwargs['n_estimators']), train_result))
    elif "xgboost" in method_list:
        _predict_result.append(fit_predict_each_output(xgb.XGBClassifier(max_depth=kwargs["max_depth"], n_estimators=kwargs['n_estimators'], nthread=kwargs["nthread"]), _binned_train_result))
    elif "xgboost_reg" in method_list:
        _predict_result.append(fit_predict_each_output(xgb.XGBRegressor(max_depth=kwargs["max_depth"], n_estimators=kwargs['n_estimators'], nthread=kwargs["nthread"]), train_result))
    elif "svr" in method_list:
        _predict_result.append(fit_predict_each_output(svm.SVR(C=kwargs["C"], epsilon=kwargs["epsilon"]), train_result))
    elif "linear_svr" in method_list:
        _predict_result.append(fit_predict_each_output(svm.LinearSVR(C=kwargs["C"], epsilon=kwargs["epsilon"]), train_result))
    else:
        assert False, "invalid method"
    return np.asarray(_predict_result)


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
        day_idx_test, day_idx_train = day_idx[-train_day_idx_partition:], day_idx[:-train_day_idx_partition]
        for tollgate_id in range(np.size(v_train, 0)):
            for direction in range(np.size(v_train, 1)):
                for i_tw, r_tw in zip(INPUT_TW, REQUIRED_TW):
                    _train = v_train[tollgate_id, direction, day_idx_train, :, :]
                    _test = v_train[tollgate_id, direction, day_idx_test, :, :]

                    _train_list = np.concatenate([_train[:, i_tw, 0], _train[:, 0, 1:]], axis=1)
                    _train_result = _train[:, r_tw, 0]
                    _test_list = np.concatenate([_test[:, i_tw, 0], _test[:, 0, 1:]], axis=1)

                    _test_result = _test[:, r_tw, 0]
                    _test_result = sum_tws(_test_result, inputs["num_sum"])

                    _predict_result = predict(_train_list, _train_result, _test_list, method_list, **kwargs, from_bins_idx=from_bins_idx, to_bins_idx=to_bins_idx)
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
        p = None
        for tollgate_id in range(np.size(v_train, 0)):
            for direction in range(np.size(v_train, 1)):
                for i_tw, r_tw in zip(INPUT_TW, REQUIRED_TW):
                    _train = v_train[tollgate_id, direction, :, :, :]
                    _test = v_test[tollgate_id, direction, :, :, :]

                    _train_list = np.concatenate([_train[:, i_tw, 0], _train[:, 0, 1:]], axis=1)
                    _train_result = _train[:, r_tw, 0]
                    _test_list = np.concatenate([_test[:, i_tw, 0], _test[:, 0, 1:]], axis=1)

                    _predict_result = predict(_train_list, _train_result, _test_list, method_list, **kwargs, from_bins_idx=from_bins_idx, to_bins_idx=to_bins_idx)
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


def test_method(rounds, method_list, inputs, parallel=False, **kwargs):
    if parallel:
        future_list = []
        for i in range(rounds):
            future_list.append(executor.submit(run, method_list, inputs, **kwargs))
        mape_list = list([item.result() for item in future_list])
    else:
        mape_list = list([run(method_list, inputs, **kwargs) for i in range(rounds)])
    log("Average MAPE of %s = " % str(method_list), np.mean(mape_list))


def main():
    rounds = 1
    num_bin = 26
    params = {
        "k": 7,
        "max_depth": 5,
        "n_estimators": 10000,
        "n_jobs": 4,
        "nthread": 4,
        "C": 0.1,  # SVR
        "epsilon": 0,  # SVR
        "par_C": 1.0,
        "par_eps": 0.01,
    }
    tw_seconds = 60 * 10

    v_train, v_test, all_ids, all_days, volume_bins = prepare_data(num_bin, time_window_seconds=tw_seconds,
                                                                   refresh=True)

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

    # test_method(rounds, ["knn"], inputs, **params)
    # test_method(rounds, ["svr"], inputs, **params)
    # test_method(rounds, ["linear_svr"], inputs, **params)
    # test_method(rounds, ["dt"], inputs, **params)
    # test_method(rounds, ["rf"], inputs, **params)
    # test_method(rounds, ["rf_reg"], inputs, **params)
    # test_method(rounds, ["rf", "linear_svr", "par"], inputs, **params)
    test_method(rounds, ["linear_svr", "par"], inputs, **params)
    test_method(rounds, ["linear_svr"], inputs, **params)
    test_method(rounds, ["rf_reg", "linear_svr", "par"], inputs, **params)
    test_method(rounds, ["par"], inputs, **params)
    # test_method(rounds, ["current"], inputs, **params)
    # test_method(rounds, ["average"], inputs, **params)
    # test_method(rounds, ["adaboost"], inputs, **params)
    # test_method(rounds, ["ridge"], inputs, **params)
    # test_method(rounds, ["linear"], inputs, **params)
    # test_method(rounds, ["huber"], inputs, **params)
    # test_method(rounds, ["lasso"], inputs, **params)
    # test_method(rounds, ["par"], inputs, **params)
    # test_method(rounds, ["theilsen"], inputs, parallel=False, **params)
    # test_method(rounds, ["ridge_reg"], inputs, **params)
    # test_method(rounds, ["ridge_reg", "linear_svr"], inputs, **params)
    # test_method(rounds, ["linear_svr", "knn"], inputs, **params)
    # test_method(rounds, ["linear_svr", "par"], inputs, **params)
    # test_method(rounds, ["linear_svr", "knn", "current"], inputs, **params)
    # test_method(rounds, ["knn", "current"], inputs, **params)
    # test_method(rounds, ["par", "current"], inputs, **params)
    # test_method(rounds, ["par", "linear_svr"], inputs, **params)
    # test_method(rounds, ["par", "linear_svr", "knn"], inputs, **params)
    # test_method(rounds, ["par", "linear_svr", "knn", "current"], inputs, **params)
    # test_method(rounds, ["xgboost"], inputs, **params)
    # test_method(rounds, ["xgboost_reg"], inputs, **params)
    # test_method(rounds, ["ridge_reg", "linear", "theilson"], inputs, **params)
    # test_method(rounds, ["dt_reg"], inputs, **params)
    # test_method(rounds, ["rf_reg"], inputs, **params)

    # run(["ridge_reg", "linear_svr"], inputs, False, **params)
    # run(["dt"], inputs, False, **params)
    # run(["ridge_reg"], inputs, False, **params)
    # run(["par"], inputs, False, **params)
    # run(["knn"], inputs, False, **params)
    # run(["rf"], inputs, False, **params)
    # run(["rf_reg"], inputs, False, **params)
    # run(["knn", "dt"], inputs, False, **params)
    # run(["knn", "linear_svr"], inputs, False, **params)
    # run(["rf", "linear_svr", "par"], inputs, False, **params)
    # run(["linear_svr", "rf_reg"], inputs, False, **params)
    run(["rf_reg", "linear_svr", "par"], inputs, False, **params)
    # run(["current"], inputs, False, **params)


if __name__ == '__main__':
    EPS = 1e-8
    with ThreadPoolExecutor(max_workers=4) as executor:
        main()
