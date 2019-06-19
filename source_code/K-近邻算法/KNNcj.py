#! /usr/bin/env python
# coding=utf-8
#from __future__ import division
import math
import time

#===============================================================================
# 求两个特征向量欧式距离
#===============================================================================
# def euclidea_distance(trains,tests):
#    words= set(tests.keys()+trains.keys())
#    tmp= 0
#    for word in words:
#        tmp+=(tests.get(word,0)-trains.get(word,0))**2
#    distance=math.sqrt(tmp)
#    return distance
#===============================================================================
# 相似度算法
#===============================================================================
def cos(source, target):
    '''
    
    :param source:
    :param target:
    '''
    numerator = sum(source[word] * target[word] for word in source if word in target)#向量内积
    sourceLen = math.sqrt(sum([value * value for value in source.values()]))#模
    targetLen = math.sqrt(sum([value * value for value in target.values()]))#模
    
    denominator = sourceLen * targetLen
    if denominator == 0:
        return 0
    else:
        return numerator / denominator#返回cosine
#===============================================================================
# 相似度矩阵
#===============================================================================
def similarVector(trains, tests):
    '''
    
    :param trains:
    :param tests:
    '''
    trainLen = len(trains)
    testsLen = len(tests)    
    simVector = {}  # 字典中存放测试样本与训练样本的相似矩阵，键值为测试样本的文件序号
    for testId in range(testsLen):
#        distance_list=[]#存放个测试样本文件与所有训练样本文件的欧式距离
        cosineList = []
        for trainId in range(trainLen):
#            distance=euclidea_distance(trains[trainId].words,tests[testId].words)#求两个测试与训练样本特征向量的欧氏距离
#            distance_list.append((distance,trainId))            
            cosine = cos(trains[trainId].words, tests[testId].words)
            cosineList.append((cosine, trainId))#放的是一个列表，里面是相似度，训练样本的编号
#         distance_list.sort()    #按照距离远近进行排序
        cosineList.sort(reverse=True)
#         simVector[testId]=distance_list    #将一个测试样本文件与所有训练样本文件的欧式距离放到字典里    
        simVector[testId] = cosineList
    return simVector

#===============================================================================
# k近邻算法情感分类
#===============================================================================
def classify(trains, tests, k):
    '''
    
    :param trains:
    :param tests:
    :param k:
    '''
    simVector = similarVector(trains, tests)#字典中存放着测试样本与训练样本的相似矩阵
    sumAccurate = 0
    # 根据相似度算法猜测测试样本文件极性，并判断正确率
    for i in range(len(tests)):
        sumPositive = 0  # positive极性总数
        sumNegative = 0  # negative极性总数
        for flag in range(k):  # 选取k个最邻近的训练样本文件，并统计极性
            index = simVector[i][flag][1]
            if trains[index].polarity:
                sumPositive += 1
            else:
                sumNegative += 1
        if sumPositive > sumNegative:  # 先判断训练样本的极性，从而猜测测试样本文件的极性
            if tests[i].polarity:  # 根据猜测的测试样本文件极性，判断正确率
                sumAccurate += 1
        else:
            if not tests[i].polarity:
                sumAccurate += 1
    acc = sumAccurate / len(tests)  # 求正确率
   # print("\nThe accuracy is: %.2f" % (acc * 100) + "%")
    print ("\nthe accuracy is:%0.2f"%(acc*100)+"%")

class Document(object):
    def __init__(self, polarity, words):   #定义构造器
        self.polarity = polarity
        self.words = words


#===============================================================================
# 从指定路径的文件中读取文档信息，并指明文件代表的极性
#===============================================================================
def readFromFile(path, polarity):
    input = open(path, 'rb')
    documents = []  # 文件列表即文档
    for line in input:
        pieces = line.split()
        pieces=[an for an in pieces if an.isalpha()]
        words = {}  # 文件中的单词对应的词频度,字典
        for piece in pieces:
            word = piece
            word=piece.lower()
            words[word] = words.get(word, 0)+1  
        if len(words)>0:
            documents.append(Document(polarity, words))  # 通过词频度和极性构造文档，并存入文档列表中
    return documents


#===============================================================================
# 域
#===============================================================================
def createDomain():
    pos = readFromFile(r'D:\corpus\kitchen\positive.review', True)
    neg = readFromFile(r'D:\corpus\kitchen\negative.review', False)
    return pos, neg



if __name__ == '__main__':
    domain = createDomain()
    trains = domain[0][200:1000] + domain[1][200:1000]  # 训练样本文档
    tests = domain[0][:200] + domain[1][:200]  #    测试样本文档
    classify(trains, tests, 20)  # 情感分类
    