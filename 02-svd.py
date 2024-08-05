# SVD 算法使用numpy里面的SVD实现
from numpy import *
from numpy import linalg as la


def loadExData():
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 欧式距离
def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


# 皮尔逊相关系数 距离
def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


# 余弦 相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# #numpy 自带的SVD矩阵分解
# Data=loadExData()
# U,Sigma,VT=la.svd(Data)
#
# print(U)
#
# print("sigma:",diag(Sigma[:2]))
#
# print("VT:",VT)
# print("result:")
# a=mat([[ 9.64365076 , 0.        ],[ 0.          ,5.29150262]])
# print(U[:,:2]*mat(diag(Sigma[:2]))*VT[:2,:])

def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        # 寻找两个用户都评级的物品
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, \
                                      dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], \
                                 dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 实际的数据集会更加的稀疏，需要降维度,用于替换对standEst的调用
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    # 建立对角矩阵,Z这里面只取前4个对角向量
    Sig4 = mat(eye(4) * Sigma[:4])  # arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  # create transformed items
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, \
                             xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=svdEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # find unrated items
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


# 遇到的问题：对于上述数据集，是否包含SVD分解在效率上没有太大的
# 区别。但是在更大规模的数据集上，SVD分解会降低程序的速度。SVD分解可以在程序调入时运
# 行一次。在大型系统中，SVD每天运行一次或者其频率更低，并且还要离线运
# #例子：使用SVD计算
# 在 NumPy 中，asmatrix 函数用于将输入转换为矩阵对象。
myDat = mat(loadExData())
result = recommend(myDat, 2, estMethod=svdEst)
print(result)
