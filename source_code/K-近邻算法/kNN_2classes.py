#! /usr/bin/env python
# coding=utf-8
from __future__ import division
import math
import codecs
#from string import maketrans

class Document(object):
    def __init__(self, polarity, words):
        self.polarity = polarity
        self.words = words

def readFromFile(path,polarity):
	docs = []
	f = codecs.open(path,'r','utf-8')
	for line in f:
		#line = line.lower()
		pieces = set(line.split())
		words = {}
		for piece in pieces:
			words[piece] = words.get(piece,0) + 1
		if len(words) > 0:
			docs.append(Document(polarity,words))
	return docs

def createDomain():
	positive = readFromFile(r'F:/new_begin/corpus/movie/positive.review',True)
	negative = readFromFile(r'F:/new_begin/corpus/movie/negative.review',False)
	#positive = readFromFile(r'F:/new_begin/corpus_7classes/1.txt',True)
	#negative = readFromFile(r'F:/new_begin/corpus_7classes/2.txt',False)
	return positive,negative

def cos(source,target):
	numerator = sum([source[word] * target[word] for word in source if word in target])
	sourceLen = math.sqrt(sum([value * value for value in source.values()]))
	targetLen = math.sqrt(sum([value * value for value in target.values()]))
	denominator = sourceLen * targetLen
	if denominator == 0:
		return 0
	else:
		return numerator / denominator

def createSimilarVector(trains,tests):
	trainLen = len(trains)
	testLen = len(tests)
	simVector = []
	for i in range(testLen):
		cosList = []
		for j in range(trainLen):
			cosine = cos(trains[j].words,tests[i].words)
			cosList.append((cosine,trains[j].polarity))
		cosList.sort(reverse=True)
		simVector.append(cosList)
	return simVector

def classify(trains,tests,k):
	simVector = createSimilarVector(trains,tests)
	testLen = len(tests)
	sumCorrect=0
	for i in range(testLen):
		sumPos = 0
		sumNeg = 0
		for j in range(k):
			if simVector[i][j][1]:
				sumPos += 1
			else:
				sumNeg += 1
		if sumPos > sumNeg:
			judge = True
		else:
			judge = False
		if judge == tests[i].polarity:
			sumCorrect += 1
	acc = sumCorrect / testLen
	print ("\n The accuracy is : %.2f%%\n" %(acc * 100))

def main():
	domain = createDomain()
	trains = domain[0][100:] + domain[1][100:]
	tests = domain[0][:100] + domain[1][:100]
	classify(trains,tests,20)

if __name__ == '__main__':
	main()