"""
file: tree.py
author: 1851995刘佳航
基于信息熵进行划分选择的决策树算法,无剪枝
"""

import numpy as np
import copy
from graphviz import Digraph
import json



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


# 获取所有节点中最多子节点的叶节点
def getMaxLeafs(myTree):
    numLeaf = len(myTree.keys())
    for key, value in myTree.items():
        if isinstance(value, dict):
            sum_numLeaf = getMaxLeafs(value)
            if sum_numLeaf > numLeaf:
                numLeaf = sum_numLeaf
    return numLeaf


def plot_model(tree, name):
    g = Digraph("G", filename=name, format='png', strict=False)
    first_label = list(tree.keys())[0]
    g.node("0", first_label)
    _sub_plot(g, tree, "0")
    leafs = str(getMaxLeafs(tree) // 10)
    g.attr(rankdir='LR', ranksep=leafs)
    g.view()


root = "0"


def _sub_plot(g, tree, inc):
    global root

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            g.node(root, list(tree[first_label][i].keys())[0])
            g.edge(inc, root, str(i))
            _sub_plot(g, tree[first_label][i], root)
        else:
            root = str(int(root) + 1)
            g.node(root, tree[first_label][i])
            g.edge(inc, root, str(i))



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
    if length < 1:
        return 0.0
    _result = 0.0
    for i in range(2):
        p = _label.count(i) / length
        if p <= 0:
            return 0.0
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
    for i in range(len(_data_label)):
        _result -= (len(_each_data[i]) / _length) * EntD(_each_data[i])
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


def CGain(density, _label):
    # 密度连续值求最优
    # 密度划分点候选值
    density_data = []
    # density = data[:, 6]
    density = np.array(density, dtype="float32")
    density1 = np.sort(density)
    for i in range(len(_label) - 1):
        density_data.append((density1[i] + density1[i + 1]) / 2)
    print(density_data)
    # 求每个划分点的信息增益
    density_result = []
    for x in density_data:
        density_result.append(ContinuationGain(density, _label, x))
    print(density_result)
    # 求最优增益点
    density_value, density_point = MaxIndex(density_result)
    density_point = density[density_point]
    print(density_value)
    return density_value, density_point


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
    :param _data:ndarray
    :param _label:
    :param _attribute:
    :param _attribute_label:
    :return:
    """
    _length = len(_attribute_label)
    _value = []
    for i in range(_length):
        if _attribute_label[i] == "密度":
            _value.append(CGain(_data[:, i], _label)[0])
            p = CGain(_data[:, i], _label)[1]
        else:
            _value.append(Gain(_data[:, i], _label, _attribute[i]))
    _index = MaxIndex(_value)[1]
    if _attribute_label[_index] == "密度":
        _attribute.append([-p, p])
    return _attribute_label[_index], _index, _attribute


def TreeGenerate(_data, _label, _attribute, _attribute_label):
    if _label.count(1) == 0:
        return "坏瓜"
    elif _label.count(0) == 0:
        return "好瓜"
    if _attribute_label is None or CheckSame(_data, _attribute_label):
        if _label.count(1) > _label.count(0):
            return "好瓜"
        else:
            return "坏瓜"
    _best_attribute, _index, _attribute = BestAttribute(_data, _label, _attribute, _attribute_label)
    tree_node = {_best_attribute: {}}  # 创建一个树（字典），bestFeatLabel为根结点
    # _index = 1  # 最优属性的序号，待修改
    for x in _attribute[_index]:
        _son = []
        _son_label = []
        if _best_attribute == "密度":
            midu = float(x)
            x = str(x)
            if midu < 0:
                for i in range(len(_label)):
                    if float(_data[i, _index]) <= -midu:
                        _son.append(_data[i, :])
                        _son_label.append(_label[i])
            else:
                for i in range(len(_label)):
                    if float(_data[i, _index]) >= midu:
                        _son.append(_data[i, :])
                        _son_label.append(_label[i])
        else:
            for i in range(len(_label)):
                if x == _data[i, _index]:
                    _son.append(_data[i, :])
                    _son_label.append(_label[i])
                # _data = np.delete(_data, i, axis=0)
        # _son = np.array(_son)
        # print("son:", _son)
        if not _son:
            if _son_label.count(1) > _son_label.count(0):
                tree_node[_best_attribute][x] = "好瓜"
            else:
                tree_node[_best_attribute][x] = "坏瓜"
            return tree_node
            # tree_node[_best_attribute][x] = 1  # 最多的类，未写
        else:
            # 写去掉a*
            _son = np.array(_son)
            # print("son:", _son)
            _son = np.delete(_son, _index, axis=1)
            # _data = np.delete(_data, i, axis=0)
            _attribute1 = copy.deepcopy(_attribute)
            del _attribute1[_index]
            # _attribute = np.delete(_attribute, _index, axis=0)
            _attribute_label1 = copy.deepcopy(_attribute_label)
            del _attribute_label1[_index]
            # _attribute_label = np.delete(_attribute_label, _index)
            tree_node[_best_attribute][x] = TreeGenerate(_son, _son_label, _attribute1, _attribute_label1)
    return tree_node


if __name__ == "__main__":
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
    print(data)
    label = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # 属性集a
    attribute = [["青绿", "乌黑", "浅白"],
                 ["蜷缩", "稍蜷", "硬挺"],
                 ["浊响", "沉闷", "清脆"],
                 ["清晰", "稍糊", "模糊"],
                 ["凹陷", "稍凹", "平坦"],
                 ["硬滑", "软粘"]]
    # attribute = np.array(attribute, dtype="str")
    print(attribute)
    attribute_label = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度"]
    # attribute_label = np.array(attribute_label)
    tree = TreeGenerate(data, label, attribute, attribute_label)
    # print(TreeGenerate(data, label, attribute, attribute_label))
    plot_model(tree, "test.gv")
    # print(json.dumps(tree, indent=1))

# def CGain(density, _label):
#     # 密度连续值求最优
#     # 密度划分点候选值
#     density_data = []
#     # density = data[:, 6]
#     density = np.array(density, dtype="float32")
#     density1 = np.sort(density)
#     for i in range(len(_label) - 1):
#         density_data.append((density1[i] + density1[i + 1]) / 2)
#     print(density_data)
#     # 求每个划分点的信息增益
#     density_result = []
#     for x in density_data:
#         density_result.append(ContinuationGain(density, _label, x))
#     print(density_result)
#     # 求最优增益点
#     density_value, density_point = MaxIndex(density_result)
#     density_point = density[density_point]
#     print(density_value)
#     return density_value, density_point
