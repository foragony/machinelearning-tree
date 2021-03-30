"""
file: tree.py
author: 1851995刘佳航
基于信息熵进行划分选择的决策树算法,无剪枝
"""

import numpy as np


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
决策树数据结构构造
可以利用字典构造决策树
思路如下
先找到最优的划分属性，然后这样构造字典
treenode = {最佳属性：{里面是属性的各个取值}}
本来想利用类来构造决策树，但是怎么构造没有思路，就去网上找了字典的构造方法
"""

def MaxIndex(_data):
    _max = _data[0]
    _index = 0
    for i in range(len(_data)):
        if _max < _data[i]:
            _max = _data[i]
            _index = i
    return _max, _index


def EntD(_label):
    """
    计算信息熵
    :param _label: list
    :return: double
    """
    length = len(_label)
    _result = 0.0
    for i in range(2):
        p = _label.count(i) / length
        _result -= p * np.log2(p)
    return _result


def Gain(_data, _label, _data_label):
    _length = len(_label)
    _result = 0.0
    _each_data = []
    for i in range(len(_data_label)):
        _each_data.append([])
    for i in range(_length):
        for j in range(len(_data_label)):
            if _data_label[j] == _data[i]:
                _each_data[j].append(_label[i])
                break
    _result = EntD(_label)
    for i in range(_data_label):
        _result -= (len(_each_data[i])/_length)*EntD(_each_data[i])
    return _result


def ContinuationGain(_data, _label, _value):
    """
    :param _value: double
    :param _data:  ndarray
    :param _label:  list
    :return:
    """
    _length = len(_label)
    _result = 0.0
    _data0 = []
    _label0 = []
    _data1 = []
    _label1 = []
    count = 0
    for i in range(_length):
        if _data[i] < _value:
            count += 1
            _data0.append(_data[i])
            _label0.append((_label[i]))
        else:
            _data1.append(_data[i])
            _label1.append((_label[i]))
    _result = EntD(_label) - (count / _length) * EntD(_label0) - (1 - count / _length) * EntD(_label1)
    return _result


def CheckSame(_data, _attribute_label):
    """

    :param _data: ndarry
    :param _attribute_label:
    :return:
    """
    _length = len(_attribute_label)
    for i in range(_length - 1):
        if (_data[i, :] == _data[i + 1, :]).all():
            pass
        else:
            return False
    return True


def BestAttribute(_data, _label, _attribute, _attribute_label):
    """
    选出最佳的属性
    :param _data:
    :param _label:
    :param _attribute:
    :param _attribute_label:
    :return:
    """


def TreeGenerate(_data, _label, _attribute, _attribute_label):
    if _label.count(1) == 0:
        return 0
    elif _label.count(0) == 0:
        return 1
    if _attribute_label is None or CheckSame(_data, _attribute_label):
        if _label.count(1) > _label.count(0):
            return 1
        else:
            return 0
    for i in range(len(_attribute_label)):



data = [["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697],
        ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.774],
        ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.634],
        ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.608],
        ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.556],
        ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.403],
        ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 0.481],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 0.437],
        ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0.666],
        ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0.243],
        ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0.245],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0.343],
        ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0.639],
        ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0.657],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.360],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0.593]
        ]
data = np.array(data)
label = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# 属性集a
attribute = [["青绿", "乌黑", "浅白"],
     ["蜷缩", "稍蜷", "硬挺"],
     ["浊响", "沉闷", "清脆"],
     ["清晰", "稍糊", "模糊"],
     ["凹陷", "稍凹", "平坦"],
     ["硬滑", "软粘"]]
attribute_label = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度"]

# 密度连续值求最优
# 密度划分点候选值
density = []
for i in range(len(data) - 1):
    density.append((data[i][6] + data[i + 1][6]) / 2)
# 把密度单独拿出来放到一个列表里
density_data = data[:, 6]
# for i in range(len(data)):
#     density_data.append(data[i][6])
# 求每个划分点的信息增益
density_result = []
for x in density:
    density_result.append(ContinuationGain(density_data, label, x))
# 求最优增益点
density_value, density_point = MaxIndex(density_result)
density_point = density[density_point]
