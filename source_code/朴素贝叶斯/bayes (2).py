# coding=utf-8
from __future__ import division
import math
from numpy import *

#===================================================================
#去除语料中的空格
#===================================================================
def textParse(line):
    pieces = line.strip().split()
    pieces = [piece for piece in pieces if piece.isalpha()]
    return [piece.lower() for piece in pieces]

#===================================================================
#从指定路径读取文件，返回词表与极性列表
#===================================================================
def readFromFile(path, polarity):
	fp = open(path, 'rb')
	allLines = fp.readlines()
	docs = []
	labelClass = []
	for line in allLines:
		line = textParse(line)
		words = []
		for word in line:
			words.append(word)
		docs.append(words)
		labelClass.append(polarity)
	return docs, labelClass

#==================================================================
#创建一个包含所有文档中出现的不重复词及其序号的字典
#==================================================================
def createVocabList(documents):
	vocabSet = set([]) #集合中的元素不重复
	for doc in documents:
		vocabSet = vocabSet | set(doc)
	vocabDict = dict([(word, i) for i, word in enumerate(vocabSet)])
	return vocabDict

#==================================================================
#输入是词汇表和某文档，输出是文档向量
#向量的每一个元素为0或1，表示词汇表中的单词是否出现在输入文档中
#==================================================================
def setOfWords2Vec(vocabDict, inputSet):
	martix = []
	for inputLine in inputSet:
		returnVec = [0] * len(vocabDict)
		for word in inputLine:
			if word in vocabDict:
				returnVec[vocabDict[word]] = 1
#				returnVec[vocabList.index(word)] = 1
		martix.append(returnVec)
	return martix

#=================================================================
#计算每个类别的概率，以及类别下每个特征属性划分的条件概率
#=================================================================
def trainNB0(trainMartix, trainCategory):
	numTrainDocs = len(trainMartix)
	numOfWords = len(trainMartix[0])
#	numOfWords = trainMartix.shape[1]
#   numOfWords = shape(trainMartix)[1]
	pAbusive = sum(trainCategory) / numTrainDocs #因类别标记为0和1，故sum(trainCategory)即为类别为1的文档数
	p0Num = ones(numOfWords)
	p1Num = ones(numOfWords) #Laplace校准，避免概率为0的情况
	p0Denom = 2
	p1Denom = 2
	for i in range(numTrainDocs):
		if trainCategory[i] == 0:
			p0Num += trainMartix[i] #向量加法
			p0Denom += sum(trainMartix[i]) #整行元素相加
		else:
		    p1Num += trainMartix[i]
		    p1Denom += sum(trainMartix[i])
	p0Vec = log(p0Num / p0Denom)
	p1Vec = log(p1Num / p1Denom) #避免下溢出，采用log
	return pAbusive, p0Vec, p1Vec

#=================================================================
#计算测试文档对应的概率，确定测试文档的类别
#=================================================================
def classifyNB(test, pAbusive, p0Vec, p1Vec):
	p0 = sum(test * p0Vec) + log(1 - pAbusive)
	p1 = sum(test * p1Vec) + log(pAbusive)
	if p0 > p1:
		return 0
	else:
	    return 1

#=================================================================
#将测试结果与真实标签进行比较，计算准确率
#=================================================================
def testingNB():
	pos, posLabel = readFromFile(r'D:\SUDA\corpus\movie\positive.review',1)
	neg, negLabel = readFromFile(r'D:\SUDA\corpus\movie\positive.review',0)

	posTotal = len(posLabel)
	negTotal = len(negLabel)
	trains = pos[int(posTotal*0.2):] + neg[int(negTotal*0.2):] #选取pos与neg的后百分之八十作为训练集
	tests = pos[:int(posTotal*0.2)] + neg[:int(negTotal*0.2)] #选取pos与neg的前百分之二十作为测试集
	trainCategory = posLabel[int(posTotal*0.2):] + negLabel[int(negTotal*0.2):]
	testCategory = posLabel[:int(posTotal*0.2)] + negLabel[:int(negTotal*0.2)]

	vocabDict = createVocabList(trains)
	trainMartix = setOfWords2Vec(vocabDict, trains)
	testMartix = setOfWords2Vec(vocabDict, tests)

	testLen = len(testMartix)
	pAbusive, p0Vec, p1Vec = trainNB0(trainMartix, trainCategory)
	accurate = 0
	for k in range(testLen):
		predictLabel = classifyNB(testMartix[k], pAbusive, p0Vec, p1Vec)
		if predictLabel == testCategory[k]:
			accurate += 1
	print("The accuracy is:%.2f" % (accurate / testLen * 100) + "%")

if __name__ == '__main__':
	testingNB()