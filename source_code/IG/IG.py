# coding=utf-8
from __future__ import division
import numpy
import math
#import time
import operator

#=======================================================================================================
#从指定路径读取文件内容
#=======================================================================================================
def readFromFile(path, polarity):
	fp = open(path, 'rb')
	allLines = fp.readlines()
	docs = []
	for line in allLines:
		line = line.strip().split()
		line = [piece for piece in line if piece.isalpha()]
		words = []
		doc = []
		for word in line:
			word = word.lower()
			if word not in words:
				words.append(word) #对每一个文档内的词先进行了去重
		if len(words) > 0:
			doc.append(words)
			doc.append(polarity)
			docs.append(doc)
	return docs

def createDomain(domain):
	pos = readFromFile(r'D:\SUDA\corpus\%s\positive.review' % domain, True)
	neg = readFromFile(r'D:\SUDA\corpus\%s\negative.review' % domain, False)
	return pos, neg

#=======================================================================================================
#创建词字典
#=======================================================================================================
def creatVocabDict(documents):
	vocabList = []
	for doc in documents:
		vocabList.extend(doc[0])	
	vocabList = set(vocabList)
	vocabDict = dict([(word, i) for i, word in enumerate(vocabList)]) #便于根据词找到词对应的列数
	vocabDict_word = dict([(i, word) for i, word in enumerate(vocabList)]) #计算得到信息增益后便于根据序号找到对应的词
	return vocabDict, vocabDict_word

#=======================================================================================================
#将文档根据词表转为矩阵，出现即记为1
#=======================================================================================================
def doc2Mat(documents, vocabDict):
	lenOfDocs = len(documents)
	numOfWords = len(vocabDict)
	docMat = numpy.zeros((lenOfDocs, numOfWords))
	for i in range(lenOfDocs):
		for word in documents[i][0]:
			if word in vocabDict:
				docMat[i][vocabDict[word]] = 1
	return docMat			

#=======================================================================================================
#创建类别矩阵，记录输入文档对应类别
#=======================================================================================================
def cateVect(documents):
	cateVect = [0] * len(documents)
	for i in range(len(documents)):
		if documents[i][1]:
			cateVect[i] = 1
		else:
			cateVect[i] = 0
	return numpy.array(cateVect) #将列表转为矩阵

#=======================================================================================================
#计算经验熵
#=======================================================================================================
def entropy(cateVect):
	pC1 = numpy.sum(cateVect) / len(cateVect)
	return pC1, pC1 * numpy.log2(pC1) * (-1) - (1 - pC1) * numpy.log2(1 - pC1)

#=======================================================================================================
#计算条件熵
#=======================================================================================================
def expectedEntropy(docMat, cateVect):
	pC1 = entropy(cateVect)[0]
	lenOfDocs = len(docMat)
	wordsNum = len(docMat[0])
	pT = numpy.zeros(wordsNum)
	pT_C1 = numpy.zeros(wordsNum)
	pT_C0 = numpy.zeros(wordsNum)

	for i in range(lenOfDocs):
		pT += docMat[i] #该词出现的文档数
		if cateVect[i]:
			pT_C1 += docMat[i] #该词出现且类别为pos的文档数
		else:
			pT_C0 += docMat[i] #该词出现且类别为neg的文档数

	pTe = pT / lenOfDocs #出现该词的文档数 / 文档总数
	pT_C1e = pT_C1 / pT #该词出现且类别为1的文档数 / 该词出现的文档数
	pT_C0e = pT_C0 / pT #pT_C0e = numpy.ones(wordsNum) - pT_C1e

#	method 2
#	pT = pT / lenOfDocs
#	pT_C1 = pT_C1 / numpy.sum(cateVect)
#	pT_C1e = (pT_C1 * pC1) / pT
#	pT_C0e = numpy.ones(wordsNum) - pT_C1e
#	pT_C0 = pT_C0 / numpy.sum(cateVect)
#	pT_C0e = (pT_C0 * (1 - pC1)) / pT

	log_pT_C1e = numpy.zeros(wordsNum)
	for i, item in enumerate(pT_C1e):
		if item != 0.0:
			log_pT_C1e[i] = numpy.log2(item) #防止概率为0无法用log进行计算
	log_pT_C0e = numpy.zeros(wordsNum)
	for i, item in enumerate(pT_C0e):
		if item != 0.0:
			log_pT_C0e[i] = numpy.log2(item)
	#p1 = pTe * (pT_C1e * log_pT_C1e * (-1) - pT_C0e * log_pT_C0e)
	p1 = - pTe * (pT_C1e * log_pT_C1e + pT_C0e * log_pT_C0e)
	pTne = numpy.ones(wordsNum) - pTe
	pT_C1ne = (numpy.tile(numpy.sum(cateVect), wordsNum) - pT_C1) / (numpy.tile(lenOfDocs, wordsNum) - pT) #numpy.sum(cateVect)求得类别为1的文档数，使用tile是为了将其扩展为对应维数的矩阵方便运算
#	pT_C1ne = (numpy.array([numpy.sum(cateVect)] * wordsNum) - pT_C1) / (numpy.array([lenOfDocs] * wordsNum) - pT)
	pT_C0ne = numpy.ones(wordsNum) - pT_C1ne
	log_pT_C1ne = numpy.zeros(wordsNum)
	for i, item in enumerate(pT_C1ne):
		if item != 0.0:
			log_pT_C1ne[i] = numpy.log2(item)
	log_pT_C0ne = numpy.zeros(wordsNum)
	for i, item in enumerate(pT_C0ne):
		if item != 0.0:
			log_pT_C0ne[i] = numpy.log2(item)
	p2 = pTne * (pT_C1ne * log_pT_C1ne * (-1) - pT_C0ne * log_pT_C0ne)

	return p1 + p2

if __name__ == '__main__':
	domain = createDomain('kitchen')
	trains = domain[0][:] + domain[1][:]
	#domain = [[], []]
	#domain[0], domain[1] = createDomain('kitchen')
	#trains = domain[0][:] + domain[1][:]
	vocabDict, vocabDict_word = creatVocabDict(trains)
	docMat = doc2Mat(trains, vocabDict)
	cateVect = cateVect(trains)
	gainVect = entropy(cateVect)[1] - expectedEntropy(docMat, cateVect)
	gainList = [(gainVect[i], i, vocabDict_word[i]) for i in range(len(gainVect))]
	print(len(gainList))
	gainList = sorted(gainList, key = operator.itemgetter(0), reverse = True)
	print (gainList[:10])