import torch
from IPython import display
from d2l import torch as d2l


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 求每行的最大值，即每个样本中对10个类别预测概率中最大的概率
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def sum_list(list):
    sum = 0
    for i in list:
        sum += i
    return sum


# 计算指定数据集上正确率
def data_accuracy(net, data_iter):
    acc = []
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            acc.append(accuracy(net(X), y))
            total += len(y)
    return sum_list(acc) / total


def train_module_of_torch(net, data_iter, test_iter, loss, updater, epoch):
    for i in range(epoch):
        loss_value = []
        acc_value = []
        num_of_sample = 0
        for X, y in data_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            # 损失累加
            loss_value.append(float(l.sum()))
            acc_value.append(accuracy(y_hat,y))
            num_of_sample += len(y)
        sum_of_loss = sum_list(loss_value)
        sum_of_acc = sum_list(acc_value)
        print(data_accuracy(net, test_iter), sum_of_loss / num_of_sample, sum_of_acc / num_of_sample)
