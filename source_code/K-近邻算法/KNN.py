# -*- coding: utf-8 -*-

from numpy import *  # 导入
import operator  # 导入运算符模块
from os import listdir  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表


def createDataSet():  # 创建数据集和标签
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# K-近邻分类算法
def classify0(inX, dataSet, labels, k):  # 用于分类的输入向量inX，输入的训练样本集，标签，计算的距离
    dataSetSize = dataSet.shape[0]  # 数据集的宽
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 输入的向量和数据集中各向量相减
    sqDiffMat = diffMat ** 2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 求上面数据的和，行相加
    diatances = sqDistances ** 0.5  # 开方
    sortedDistIndicies = diatances.argsort()  # 取输入向量和数据集中的距离的从小到大排序的下标
    classCount = {}  # 定义一个空字典
    for i in range(k):  # 取计算欧式前K个小的值
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 若字典键的值有voteIlabel,则返0
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)  # 排序，返回频率最高的元素标签
    return sortedClassCount[0][0]


# 将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  # 得到文件的每行数据的列表
    numberOfLines = len(arrayOLines)  # 得到文件的行数（即有多少样本）
    returnMat = zeros((numberOfLines, 3))  # 返回一个文件的行数为行数，3列的0矩阵
    classLabelVector = []  # 创建一个标签向量的列表
    index = 0  # 索引为0
    for line in arrayOLines:  # 循环处理文件中的每行数据，解析文件数据到列表
        line = line.strip()  # 截取掉首尾回车字符
        listFromLine = line.split('\t')  # 利用\t分离字符串
        returnMat[index, :] = listFromLine[0:3]  # 选取样本前3个元素，存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  # 将每行样本的最后一个元素存储到向量中
        index += 1
    return returnMat, classLabelVector  # 返回矩阵和向量


# 将数字特征值转化为0-1
def autoNorm(dataSet):  # 输入为数据集数据
    minVals = dataSet.min(0)  # 存放每列的最小值
    maxVals = dataSet.max(0)  # 存放每列的最大值
    ranges = maxVals - minVals  # 函数计算可能的取值范围，并创建新的返回矩阵
    normDataSet = zeros(shape(dataSet))  # 初始化归一化数据集，复制成输入矩阵同样大小的矩阵
    m = dataSet.shape[0]  # 行
    normDataSet = dataSet - tile(minVals, (m, 1))  # 对矩阵dataSet每个元素求差
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 具体特征值相除
    return normDataSet, ranges, minVals  # 返回归一化矩阵，取值范围以及最小值


# 测试分类器的性能,
def datingClassTest():
    hoRatio = 0.10  # 取测试样本占数据集样本的10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 得到样本集和样本标签
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 得到归一化样本集，取值范围以及最小值
    m = normMat.shape[0]  # 样本集行数
    numTestVecs = int(m * hoRatio)  # 测试样本集数量
    errorCount = 0.0  # 初始化错误率
    for i in range(numTestVecs):  # 循环计算样本集错误率
        # 传参给分类器进行分类，每个for循环改变的参数只有第一项的测试数据而已
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # 打印当前测试数据的分类结果个真实结果
        print("the classifier came back with:%d,the real answer is :%d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0  # 如果分类结果不等于真是结果，错误计数加一
    print("the total error rate is :%f" % (errorCount / float(numTestVecs)))  # 计算错误率并输出


# 自定义分类器，输入信息得到结果，编写预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']  # 建立输出列表
    percentTats = float(input("percentage of time spent playing video games"))  # 读取键盘输入的数值
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of iceCream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 从文本文件中解析数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化特征值
    inArr = array([ffMiles, percentTats, iceCream])  # 将先前读取的键盘输入填入数组
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)  # 分类：这里也对输入数据进行了归一化
    print("you will probably like this person:", resultList[classifierResult - 1])  # 打印分类信息


############################################################
# 将图像格式化处理为一个向量 32*32图像转换为1*1024#将图像格式化处理为一个向量 32*32图像转换为1*1024向量
"""
首先创建1*1024的numpy数组
然后打开给定的文件，读出文件前32行，
将每行的头32个字符存储在numpy数组中
"""


def img2vector(filename):
    returnVect = zeros((1, 1024))  # 初始化要返回1*1024向量
    fr = open(filename)
    for i in range(32):  # 循环读取文件的前32行
        lineStr = fr.readline()
        for j in range(32):  # 每行的头32个字符存储到要返回的向量中
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect  # 返回输出的1*1024向量


# 测试算法，将数据输入到分类器，检测分类器的执行效果
# 有关读取目录下文件信息
def handwritingClassTest():
    hwLabels = []  # 初始化手写数字标签列表
    trainingFileList = listdir('trainingDigits')  # 获取训练文件目录名
    m = len(trainingFileList)  # 获取训练文件数目
    trainingMat = zeros((m, 1024))  # 初始化训练矩阵，以文件数目为行，1024列
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 从文件名解析出分类数字
        fileStr = fileNameStr.split('.')[0]  # 以。开始分割文件名，取分割后数组的第一个元素，即去掉后面的txt
        classNumStr = int(fileStr.split('_')[0])  # 以_分割，获取前面的那个数字
        hwLabels.append(classNumStr)  # 存储解析出的数字存到标签中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  # 载入图像
    testFileList = listdir('testDigits')  # 获取测试目录信息
    errorCount = 0.0  # 初始化错误计数
    mTest = len(testFileList)  # 获取测试文件数目
    for i in range(mTest):  # 开始测试
        fileNameStr = testFileList[i]  # 从文件名接卸出分类数组
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('trainingDigits/%s' % fileNameStr)  # 载入图像
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  # 参数传入分类器进行分类
        # 打印出分类结果和真实结果
        print("the classifier came back with:%d,the real answer is :%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):  # 如果分类结果不等于真实结果，节错误计数+1
            errorCount += 1.0
    print("\nthe total number of errors is :%d" % errorCount)  # 输出错误次数
    print("\nthe total error rate is :%f" % (errorCount / float(mTest)))  # 输出错误率
