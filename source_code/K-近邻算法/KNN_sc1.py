#-*- coding: utf-8 -*-
import math
import time


#两个向量的欧式距离
def distance(trains,tests):
	words= set(tests.keys())|set(trains.keys())#words里存放两个测试集的的无重复的元素键值
	tmp=0
	for word in words:
		tmp+=(tests.get(word,0)-trains.get(word,0))**2
	dis=math.sqrt(tmp)
	return dis


#求相似度矩阵
def similarVector(trains,tests):
	trainLen=len(trains)
	testLen=len(tests)
	simVector={}#存放相似矩阵，键值为测试集的序号
	for testId in range(testLen):
		distance_list=[] #存放测试集与所有样本的欧式距离
		cosList=[]#存放相似度
		for trainId in range(trainLen):
			dis=distance(trains[trainId].words,tests[testId].words)#求测试集和训练集的欧式距离
			distance_list.append((dis,trainId))
		distance_list.sort()#按照距离远近排序
		simVector[testId]=distance_list#将测试集与所有训练集的距离放在字典里
	return simVector #返回字典

#KNN
def classify0(trains,tests,k):
	simVector=similarVector(trains,tests)
	sumAccurate=0
	for i in range(len(tests)):
		sumPositive=0
		sumNegative=0
		for flag in range(k):
			index=simVector[i][flag][1]
			if trains[index].polarity:
				sumPositive+=1
			else:
				sumNegative+=1
		if sumPositive>sumNegative:
			if tests[i].polarity:
				sumAccurate+=1

		else:
			if not tests[i].polarity:
				sumAccurate+=1
	acc=sumAccurate/len(tests)
	print("\nthe accracy is :%0.2f"%(acc*100)+"%")

class Document(object):
	def __init__(self,polarity,words):
		self.polarity=polarity
		self.words=words

#读取文件信息，并说明极性
def readFromFile(path,polarity):
	input=open(path,'rb')
	documents=[]#文档列表
	for line in input:
		pieces=line.split()
		pieces=[amh for amh in pieces if amh.isalpha()]
		words={}#该字典存放文件中单词对应的频度
		for piece in pieces:
			word=piece
			word=piece.lower()
			words[word]=words.get(word,0)+1
		if len(words)>0:
			documents.append(Document(polarity,words))#根据词频度和极性构造文档，并存入文档列表
	return documents


def createDomain():
	pos=readFromFile(r'D:\SUDA\corpus\kitchen\positive.review',True)
	neg=readFromFile(r'D:\SUDA\corpus\kitchen\negative.review',False)
	return pos,neg




if __name__=='__main__':
	domain=createDomain()
	trains=domain[0][200:1000]+domain[1][200:1000]
	tests=domain[0][:50]+domain[1][:50]
	classify0(trains,tests,20)