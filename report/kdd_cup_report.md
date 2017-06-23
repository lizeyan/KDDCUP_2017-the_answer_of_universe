# KDD CUP2017

the answer of universe

李则言 陈光斌 陈禹东

---

## 最终成绩

### Travel Time Prediction

未在线评测

### Volume Prediction

MAPE： 	0.1760

Rank：	55

## 最初的尝试——直接拟合曲线

最开始的时候希望直接拟合travel time和volume曲线之前若干点到某一点的映射。

也就是说对于travel time和volume两个问题，分别训练这样的映射：

$$
<\text{previous average travel time}, \text{daily time}, \text{day of week}, \text{weather features}> \to <\text{average travel time}>
$$

$$
<\text{previous average volume}, \text{daily time}, \text{day of week}, \text{weather features}> \to <\text{average volume}>
$$

为了平滑数据以及便于使用一些分类模型来处理问题，我们将average travel time和average volume按照等深或者等宽的方法分成若干个桶，然后将训练数据和测试数据都转化为桶标签（而不是连续的值）。模型给出的结果也是桶标签，我们再将这个桶的平均值作为预测的结果。

在本地，我们使用k-fold cross validation的方法来进行测试。