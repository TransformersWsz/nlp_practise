#coding=utf-8
from __future__ import division
import math
import numpy
import codecs


class Document :
	def __init__(self,polarity,words) :
		self.polarity = polarity
		self.words = words


def readFromFile(path,polarity) :
	f = codecs.open(path,'r','utf-8')
	punctuations = '~`!@#$%^&*()-_+={[}]|\\;:\'"<,>.?/'
	v = ' ' * len(punctuations)
	table = str.maketrans(punctuations,v)
	docs = []
	for line in f :
		line = line.translate(table)
		pieces = set(line.lower().strip().split())
		pieces = [piece for piece in pieces if len(piece) > 0]
		words = {}
		for word in pieces :
			if word not in words :
				words[word] = 1
		if len(words) > 0 :
			docs.append(Document(polarity,words))
	return docs


def createDomain(domain) :
	positive = readFromFile(r'D:\SUDA\corpus\%s\positive.review' % domain,True)
	negative = readFromFile(r'D:\SUDA\corpus\%s\negative.review' % domain,False)
	return negative,positive


def createVocabDict(documents) :
	vocabList = []
	for doc in documents :
		vocabList += doc.words.keys()
	vocabList = set(vocabList)
	vocabDict = dict([(word,i) for i,word in enumerate(vocabList)])
	#vocabDict_word = dict([(i,word) for i,word in enumerate(vocabList)])
	#return vocabDict,vocabDict_word
	return vocabDict


def doc2VecMat(documents,vocabDict) :
	docsLen = len(documents)
	dictLen = len(vocabDict)
	docMat = numpy.zeros((docsLen,dictLen))
	for i in range(docsLen) :
		wordsVec = numpy.zeros(dictLen)
		for word in documents[i].words :
			if word in vocabDict :
				wordsVec[vocabDict[word]] = 1
		docMat[i] = wordsVec
	return docMat


def categoryOfDoc2Vec(documents) :
	category = [0] * len(documents)
	for i in range(len(documents)) :
		if documents[i].polarity :
			category[i] = 1
		else :
			category[i] = 0
	return numpy.array(category)


def calEntropy(category) :
	pC1 = numpy.sum(category) / len(category)
	return pC1,(pC1 * (numpy.log2(pC1)) + (1 - pC1) * (numpy.log2(1 - pC1)))


def calIG(docMat,category) :
	pC1 = calEntropy(category)[0]
	pC2 = 1 - pC1
	entropy = calEntropy(category)[1]
	docsNum = len(docMat)
	wordsNum = len(docMat[0])
	pT = numpy.zeros(wordsNum)
	pT_C1 = numpy.ones(wordsNum)
	pT_C2 = numpy.ones(wordsNum)
	for i in range(docsNum) :
		pT += docMat[i]
		if category[i] == 1 :
			pT_C1 += docMat[i]
		else :
			pT_C2 += docMat[i]
	pT = pT /docsNum
	pT_C1 = pT_C1 / numpy.sum(pT_C1)
	pT_C2 = pT_C2 / numpy.sum(pT_C2)
	pC1_T = (pT_C1 * pC1) / pT
	pC2_T = (pT_C2 * pC2) / pT
	p1 = pT * (pC1_T * numpy.log2(pC1_T) + pC2_T * numpy.log2(pC2_T))
	pTn = numpy.ones(wordsNum) - pT
	arr = numpy.ones(wordsNum)
	pC1_Tn = ((arr - pT_C1) * pC1) / pTn
	pC2_Tn = ((arr - pT_C2) * pC2) / pTn
	p2 = pTn * (pC1_Tn * numpy.log2(pC1_Tn) + pC2_Tn * numpy.log2(pC2_Tn))
	return (p1 + p2 - entropy)
	


if __name__ == '__main__':
	domain = createDomain('movie')

	trains = domain[0][200:] + domain[1][200:]
	#tests = domain[0][:200] + domain[1][:200]

	#vocabDict,vocabDict_word = createVocabDict(trains)
	vocabDict = createVocabDict(trains)

	docMat = doc2VecMat(trains,vocabDict)

	category = categoryOfDoc2Vec(trains)

	#pC1,entropy =  calEntropy(category)

	ig = calIG(docMat,category)
	ig = sorted(ig,reverse = True)
	for i in range(len(ig)) :
		print (ig[i])