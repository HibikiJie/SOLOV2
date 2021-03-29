import numpy


def iou(box, big_brothers):
    """
    计算交并比

    参数：
        box: w,h
        big_brothers: 此时选举出的老大哥

    :return: 交并比值
    """
    w_min = numpy.minimum(box[0], big_brothers[:, 0])
    h_min = numpy.minimum(box[1], big_brothers[:, 1])
    intersection = h_min * w_min
    box_area = box[0] * box[1]
    big_brother_area = big_brothers[:, 0] * big_brothers[:, 1]
    return intersection / (box_area + big_brother_area - intersection)


def k_means(boxes, k, choice_big_brother=numpy.median):
    """
    聚类步骤：
        1、首先输入k的值，即我们希望将数据集经过聚类得到k个分组。
        2、从数据集中随机选择k个数据点作为初始老大哥（质心，Centroid）
        3、对集合中每一个小弟，计算与每一个老大哥的距离（距离的含义后面会讲），离哪个老大哥距离近，就跟定哪个老大哥。
        4、这时每一个老大哥手下都聚集了一票小弟，这时候召开人民代表大会，每一群选出新的老大哥（其实是通过算法选出新的质心）。
        5、如果新大哥和老大哥之间的距离小于某一个设置的阈值（表示重新计算的质心的位置变化不大，趋于稳定，或者说收敛），
           可以认为我们进行的聚类已经达到期望的结果，算法终止。
        6、如果新大哥和老大哥距离变化很大，需要迭代3~5步骤。
    参数：
        boxes: 数据类型为array，盒子里为n维的宽和高数据
        k: 需要分类的数量
        choice_big_brother: 选择新的老大哥的方法，默认为中值
    返回：
        选出的老大哥
    """
    '''获取数据的量'''
    rows = boxes.shape[0]

    '''初始化距离'''
    distances = numpy.empty((rows, k))

    '''随机挑选老大哥'''
    big_brothers = boxes[numpy.random.choice(rows, k, replace=False)]

    '''定义最后决出的老大哥'''
    last_vote = numpy.zeros((rows,))
    count = 0
    while True:

        '''循环，计算每一个小老弟到每一个老大哥的距离，以交并比的值为投票依据'''
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], big_brothers)

        '''计票，计算所有小弟分别选择跟随哪一位老大哥'''
        vote = distances.argmin(axis=1)

        '''循环，每一个老大哥手下都聚集了一票小弟，这时候召开人民代表大会，每一群选出新的老大哥'''
        for big_brother in range(k):
            '''选出新的老大哥的方式为，每一群小弟的中值'''
            big_brothers[big_brother] = choice_big_brother(boxes[vote == big_brother], axis=0)

        '''每一个新老大的成员，如果不再变化，则结束漫长的选举'''
        if (last_vote == vote).all():
            break

        '''计算完新旧两轮的变化后，新的老大哥也变成了旧的老大哥'''
        last_vote = vote
        count += 1
        print(count)
    print('选举次数：', count)
    return big_brothers


if __name__ == '__main__':

    wh = numpy.loadtxt('wh.txt')+1
    print(wh.min())
    boxes = k_means(wh, 3)
    print(boxes)
