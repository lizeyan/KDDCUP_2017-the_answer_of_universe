from datetime import *
import numpy as np
from extract_features import *
from scipy.stats import *


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


def main():
    v_train, _, _ = extract_volume_knn("./training/volume(table 6)_training.csv", "volume_for_knn_train.npy")
    v_test, all_ids, all_days = extract_volume_knn("./testing_phase1/volume(table 6)_test1.csv", "volume_for_knn_test.npy")
    log("Train data shape:", np.shape(v_train))
    log("Test data shape:", np.shape(v_test))
    num_bins = 21
    AVERAGE_VOLUME = 0
    DAY_OF_WEEK = 1
    all_volume = np.concatenate([v_train[:, :, :, :, AVERAGE_VOLUME].flatten(), v_test[:, :, :, :, AVERAGE_VOLUME].flatten()])
    volume_bins = get_bins(num_bins - 1, all_volume[all_volume > 0])
    volume_bins = [0] + volume_bins
    print(volume_bins)

    def from_bins_idx(arr):
        return np.vectorize(volume_bins.__getitem__)(arr)

    def to_bins_idx(arr):
        return np.searchsorted(volume_bins, arr)

    INPUT_TW = ((18, 19, 20, 21, 22, 23), (45, 46, 47, 48, 49, 50))
    REQUIRED_TW = ((24, 25, 26, 27, 28, 29), (51, 52, 53, 54, 55, 56))

    #  cross_validate
    fold = 20
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
                _train_list = np.concatenate([to_bins_idx(_train[:, i_tw, 0]), _train[:, 0, 1:]], axis=1)
                _train_result = to_bins_idx(_train[:, r_tw, 0])
                _test_list = np.concatenate([to_bins_idx(_test[:, i_tw, 0]), _test[:, 0, 1:]], axis=1)
                _test_result = _test[:, r_tw, 0]
                _predict_result = from_bins_idx(knn_predict(_train_list, _train_result, _test_list))
                rst_count += len(r_tw)
                with np.errstate(divide='ignore', invalid='ignore'):
                    diff = np.abs(_predict_result - _test_result)
                    fraction = diff / _test_result
                    fraction[diff == 0] = 0
                    rst_sum += np.sum(fraction)
    log("Cross validate MAPE by KNN", rst_sum / rst_count * 2)
    #  test
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
                _predict_result = from_bins_idx(knn_predict(_train_list, _train_result, _test_list))
                for day in range(np.size(v_test, 2)):
                    for idx, tw in enumerate(r_tw):
                        timestamp = int(all_days[day]) * 86400 - 3600 * 8 + 1200 * tw
                        print("%s, [%s, %s), %s, %d" % (all_ids[tollgate_id], datetime.fromtimestamp(timestamp).isoformat(" "), datetime.fromtimestamp(timestamp + 1200).isoformat(" "), direction, np.asscalar(_predict_result[day, idx])), file=test_output_file)
    test_output_file.close()

if __name__ == '__main__':
    main()

