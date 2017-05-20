# The Answer is 42

KDD CUP 2017

# Change Log

## 2017-5-20
增加one-hot和降雨量信息，目前只是简单地根据降雨量把降雨等级分为0,1,2,3,4四个等级，见utility.rain_level()

## 2017-04-16

- 完成基本的KNN分类：
  - 将volume数据离散化为`num_bins`类
  - 使用`6:00-8:00`的6个时间窗堆的average volume和其他属性作为feature，寻找k个欧氏距离最接近的日期，将其`8:00-10:00`的数据（离散化后）的众数作为结果
- Plan：
  - 可优化的参数。模型越简单泛化能力应当越好，feature的权重和feature的数目应该尽量简单：
    - k
    - num_bins
    - feature的权重
    - feature的选择
  - 使用HMM方法测试
  - 使用RNN方法测试

## 2017-04-18

- KNN调参尝试：
  - 结果：```num_bins = 51, k =6, feature_weight = [0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 0.7, 0.9], feature = [六个时间窗的volume, day, is_holiday]```
- 尝试决策树、随机森林：
  - MAPE测试结果并不比KNN好

# Expected File Structure:

```
.
├── extract_features.py
├── pic
│   ├── day_to_average_travel_time.jpg
│   ├── day_to_volume.jpg
│   ├── dir_to_volume.jpg
│   ├── iid_to_average_travel_time.jpg
│   ├── last_travel_time_to_average_travel_time.jpg
│   ├── last_volume_to_average_travel_time.jpg
│   ├── last_volume_to_volume.jpg
│   ├── tid_to_average_travel_time.jpg
│   ├── tid_to_volume.jpg
│   ├── time_to_average_travel_time.jpg
│   └── time_to_volume.jpg
├── __pycache__
│   ├── extract_features.cpython-35.pyc
│   ├── extract_features.cpython-36.pyc
│   ├── utility.cpython-35.pyc
│   └── utility.cpython-36.pyc
├── readme.md
├── run_knn.py
├── run.py
├── testing_phase1
│   ├── trajectories(table 5)_test1.csv
│   ├── volume(table 6)_test1.csv
│   └── weather (table 7)_test1.csv
├── training
│   ├── links (table 3).csv
│   ├── routes (table 4).csv
│   ├── trajectories(table 5)_training.csv
│   ├── volume(table 6)_training.csv
│   └── weather (table 7)_training.csv
├── travel_time.data
├── travel_time_decision_tree.pdf
├── utility.py
├── volume_cleaning_result.csv
├── volume.data
├── volume_for_knn_test_all_days.pickle
├── volume_for_knn_test_all_ids.pickle
├── volume_for_knn_test.npy
├── volume_for_knn_trai_all_days.pickle
├── volume_for_knn_trai_all_ids.pickle
├── volume_for_knn_train.npy
├── volume_output.csv
├── volume_preprocessed.csv
└── weather.data
```