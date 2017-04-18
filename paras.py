__author__ = 'Victor'
import os

def test(rounds, num_bin, weight1, weight2, k, method):
    cmd = "python3 run_knn.py %d %f %f %d %s" % (num_bin, weight1, weight2, k, method)
    print("\n" + cmd)
    inf_cnt = 0
    cnt = 0
    mape_sum = 0.0

    for i in range(rounds):
        output = os.popen(cmd).read()
        res = output.splitlines()
        # for item in res:
        #     print(item)
        mapeinfo = res[-1]
        mapeinfo = mapeinfo.split(' ')
        if mapeinfo[-1] == "inf":
            inf_cnt += 1
        else:
            mape = float(mapeinfo[-1])
            mape_sum += mape
            cnt += 1
    if cnt == 0:
        print("Average MAPE: mape_cnt = 0")
    else:
        print("Average MAPE = ", mape_sum / cnt)
    # print("MAPE is inf for %d times" % inf_cnt)



rounds = 30
num_bin = 51
weight2 = 0.9
weight1 = 0.7
k = 6
test(rounds, num_bin, weight1, weight2, k, "knn")
test(rounds, num_bin, weight1, weight2, k, "dt")
test(rounds, num_bin, weight1, weight2, k, "rf")
# import numpy as np
# a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# a = np.asarray(a, dtype=int)
# print(a[:, 0])

#
# k = 6
# num_bins = 51
# feature_weight = [0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 0.7, 0.9]
