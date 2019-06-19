#-*- coding: utf-8 -*-

from numpy import *
import random
import math

'''
class Document(object):
	def __init__(self,polarity,words):
		self.polarity=polarity
		self.words=words
		'''

#读文件
def readFromFile(path,polarity):
	input=open(path,'rb')
	docs=[]
	labelClass=[]
	alllines=input.readlines()
	for line in alllines:
		pieces=set(line.lower().strip().split())
		# print (pieces)
		pieces=[piece for piece in pieces if piece.isalpha()]
		words=[]
		for piece in pieces:
			words.append(piece)
		docs.append(words)
		labelClass.append(polarity)
	return docs,labelClass
'''
读取中文语料

def readFromFile(filename, polarity):
    # punctuations = [',','.','!','?','-','(',')','"']
    # 去除文章中的一些特殊字符
    punctuations = '~`!@#$%^&*()-_+={[}]|\\;:\'"<,>.?/'
    v = ' ' * len(punctuations)
    table = str.maketrans(punctuations, v)
    with codecs.open(filename, 'r', 'utf-8') as fp:
        docs = []
        labelClass = []
        # 每读一行
        for line in fp:
            line = line.translate(table)
            pieces = set(line.lower().strip().split())
            words = []
            for piece in pieces:
                words.append(piece)
            docs.append(words)
            labelClass.append(polarity)
    return docs, labelClass

'''


#创建一个包含所有文档中出现的不重复的词的列表
def createVocabList(documents):
	vocabSet = set([]) #集合中的元素不重复
	for doc in documents:
        # 取并集得到不重复的集合
		vocabSet = vocabSet | set(doc)
    # 给每个词语一个序号，从0开始
	vocabDict = dict([(word, i) for i, word in enumerate(vocabSet)])
	return vocabDict

#词表到向量的转换
def setOfWords2Vec(vocabLsit,inputSet):#参数为所有不重复的词，输入文档
	ma=[]
	for inputline in inputSet:
		returnVec=[0]*len(vocabLsit)#创建一个与词汇表等长的向量，并设为0
		for word in inputline:
			if word in vocabLsit:
				returnVec[vocabLsit[word]]=1#############
		ma.append(returnVec)
	return ma

#朴素贝叶斯分类训练器，计算每个类的类别以及类别下特征的条件概率
def trainNB0(trainMatrix,trainCategory):#文档矩阵，类别标签向量
	numTrainDocs=len(trainMatrix)#获得训练集中文档的个数
	numWords=len(trainMatrix[0])#获得训练集中单词的个数
	pA=sum(trainCategory)/float(numTrainDocs)#计算类别为1 的文档数的概率，二分类用1-pA即可
	add = [0.3] * numWords#平滑
	p0Num=ones(numWords)+add #初始化类别为0的向量
	p1Num=ones(numWords)+add  #初始化类别为1的向量
	p0Denom=2
	p1Denom=2#初始化分母变量
	#遍历训练集中的所有文档，即遍历每条句子
	for i in range(numTrainDocs):
		if trainCategory[i]==0:#如果该行的类别属于0
			p0Num+=trainMatrix[i]#将该行加到p0中，且中次数+1
			p0Denom+=sum(trainMatrix[i])
		else:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
	# 向量除法 得到在类别0的情况下，每个词的词频:p(w1|p0),p(w2|p0),...,p(wn|p0)
	p1Vect=log(p1Num/p1Denom)
	p0Vect=log(p0Num/p0Denom)
	return pA,p0Vect,p1Vect


#朴素贝叶斯分类函数
def classifyNB(test,pA,p0Vect,p1Vect):
	 # sum中计算每句话中每个词的概率的和
	 p0=sum(test*p0Vect)+log(1-pA)
	 p1=sum(test*p1Vect)+log(pA)
	 if p0>p1:
	 	return 0
	 else:
	 	return 1

#朴素贝叶斯测试函数
def testingNB():
	pos,posLabel=readFromFile(r'D:\SUDA\corpus\electronics\positive.review',1)
	neg,negLabel=readFromFile(r'D:\SUDA\corpus\electronics\negative.review',0)
	posTotal=len(posLabel)
	negTotal=len(negLabel)
	trains=pos[50:]+neg[50:]
	tests=pos[:50]+neg[:50]

	trainCategory=posLabel[50:]+negLabel[50:]
	testCategory=posLabel[:50]+negLabel[:50]
	'''
	利用百分比取范围

	posTotal = len(posLabel)
	negTotal = len(negLabel)
	trains = pos[int(posTotal*0.2):] + neg[int(negTotal*0.2):] #选取pos与neg的后百分之八十作为训练集
	tests = pos[:int(posTotal*0.2)] + neg[:int(negTotal*0.2)] #选取pos与neg的前百分之二十作为测试集

	'''
	#创建一个包含所有词的列表
	myVocabList=createVocabList(trains)
	trainMatrix=setOfWords2Vec(myVocabList,trains)
	testMartix=setOfWords2Vec(myVocabList,tests)
	pA,p0Vect,p1Vect=trainNB0(trainMatrix,trainCategory)
	acc=0
	for k in range(len(testMartix)):
		predict=classifyNB(testMartix[k],pA,p0Vect,p1Vect)
		if predict == testCategory[k]:
			acc+=1
	print("the accuracy is :%0.2f"%(acc/len(testMartix) * 100)+"%")

if __name__=='__main__':
	testingNB()

