

import numpy as np

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
# print(data[:, 2])

# bbb = [[0],
#        [0],
#        [0]]
bbb = []
for i in range(3):
    aaa = []
    bbb.append(aaa)

print(bbb)
for i in range(3):
    # aaa = []
    bbb[i].append(525)
    bbb[i].append(55)
    bbb[i].append(2)
bbb = np.array(bbb)
x = np.delete(bbb, 1, axis=1)
print(bbb)
print(x)
print((x[0, :]==x[1, :]).all())
# print(bbb[1][1:])
# 利用函数递归创建决策树
# dataSet：数据集
# labels：标签列表，包含了数据集中所有特征的标签
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]    # 取出dataSet最后一列的数据

    if classList.count(classList[0]) == len(classList): # classList中classList[0]出现的次数=classList长度，表示类别完全相同，停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                            # 遍历完所有特征时返回出现次数最多的类别
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)        # 计算划分的最优特征（下标）
    bestFeatLabel = labels[bestFeat]                    # 数据划分的最优特征的标签（即是什么特征）
    myTree = {bestFeatLabel:{}}                         # 创建一个树（字典），bestFeatLabel为根结点
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)# 利用递归构造决策树
    return myTree
