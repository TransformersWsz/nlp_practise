#! /usr/bin/env python
# coding=utf-8
from __future__ import division
import math
import codecs

class Document(object):
    def __init__(self, polarity, words):
        self.polarity = polarity
        self.words = words

def readFromFile(path,polarity):
	docs=[]
	f = codecs.open(path,'r','utf-8')
	for line in f:
		pieces = set(line.split())
		words={}
		for piece in pieces:
			words[piece]=1
		if len(words)>0:
			docs.append(Document(polarity,words))
	return docs

def createDomain():
	e0 = readFromFile(r'F:/new_begin/corpus/1.txt',0)
	e1 = readFromFile(r'F:/new_begin/corpus/2.txt',1)
	e2 = readFromFile(r'F:/new_begin/corpus/3.txt',2)
	e3 = readFromFile(r'F:/new_begin/corpus/4.txt',3)
	e4 = readFromFile(r'F:/new_begin/corpus/5.txt',4)
	e5 = readFromFile(r'F:/new_begin/corpus/6.txt',5)
	e6 = readFromFile(r'F:/new_begin/corpus/7.txt',6)
	return e0,e1,e2,e3,e4,e5,e6

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
		enum = [0,0,0,0,0,0,0]
		for j in range(k):
			m = simVector[i][j][1]
			enum[m] += 1
		for t in range(7):
			max = 0
			if enum[t] > enum[max]:
				max = t
		if max == tests[i].polarity:
			sumCorrect += 1
	acc = sumCorrect / testLen
	print "\n The accuracy is : %.2f%%\n" %(acc * 100)

def main():
	domain = createDomain()
	trains = domain[0][90:450] + domain[1][90:450] + domain[2][90:450] + domain[3][18:90] + domain[4][90:450] + domain[5][90:450] + domain[6][90:450]
	tests = domain[0][:90] + domain[1][:90] + domain[2][:90] + domain[3][:18] + domain[4][:90] + domain[5][:90] + domain[6][:90]
	classify(trains,tests,20)

if __name__ == '__main__':
	main()