import numpy as np
import operator

def DataSet():
    group = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    lables = ['爱情片','爱情片','爱情片','动作片','动作片','动作片']
    return group,lables

def KNN(in_x, x_lables, y_lables, k):
    x_lables_size = x_lables.shape[0]
    distances = (np.tile(in_x, (x_lables_size, 1)) - x_lables) ** 2
    ad_distances = distances.sum(axis = 1)
    sq_distances = ad_distances ** 0.5
    ed_distances = sq_distances.argsort()
    classdict = {}
    for i in range(k):
        voteI_lables = y_lables[ed_distances[i]]
        classdict[voteI_lables] = classdict.get(voteI_lables, 0) + 1
    sort_classdict = sorted(classdict.items(), key=operator.itemgetter(1), reverse=True)
    return sort_classdict[0][0]

if __name__ == '__main__':
    group, lables = DataSet()
    test_x = [18,90]
    print('输入数据所对应的类别是:{}'.format(KNN(test_x, group, lables, 3)))
