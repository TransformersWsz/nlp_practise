#-*- coding:UTF-8 -*-

from numpy import *
import random
#词表到向量的转换函数

def loadDataSet():
	postingList=[['my','dog','has','flea','problems','help','please'],
					['maybe','not','take','take','him','to','dog','park','stupid'],
					['my','dalmation','is','so','cute','I','love','him'],
					['stop','posting','stupid','worthless','garbage'],
					['mr','licks','ate','my','steak','how','to','stop','him'],
					['quit','buying','worthless','dog','food','stupid']]#词条切分后的文档集合
	classVec=[0,1,0,1,0,1]#类别标签的集合 1代表侮辱的文字，0代表正常言论
	return postingList,classVec#返回切分后的文档，以及类别标签


#创建一个包含所有文档中出现的不重复词的列表
def createVocabList(dataSet):
	vocabSet=set([])#创建一个空集
	for document in dataSet:
		vocabSet=vocabSet | set(document)#将新词集合添加到创建的集合中，求并集
	return list(vocabSet)#返回一个包含所有文档中出现的不重复词的列表


#词表到向量的转换

def setofWords2Vec(vocabList,inputSet):#参数为所有不重复的词汇表，输入文档
	returnVec=[0]*len(vocabList)#创建一个与词汇表等长的向量，并且设为0
	for word in inputSet:#遍历文档中的词汇
		if word in vocabList:
			returnVec[vocabList.index(word)]+=1#如果文档中的单词在词汇表里，则对应向量位置+1
		else:
			print("the word:%s is not in my vocablary!"%word)#向量的每个元素为1或者0，表示词汇表中的单词在输入文档中是否出现
	return returnVec

#朴素贝叶斯分类器训练函数

def trainNB0(trainMatrix,trainCategory):#文档矩阵。类别标签向量
	numTrainDocs=len(trainMatrix)#获得训练集中文档个数
	numWords=len(trainMatrix[0])#获得训练集中单词的个数
	pAbusive=sum(trainCategory)/float(numTrainDocs)#计算文档属于侮辱性文档的概率
	p0Num=ones(numWords);p1Num=ones(numWords)#初始化概率的分子变量
	p0Denom=2.0;p1Denom=2.0#初始化概率的分母变量
	for i in range(numTrainDocs):#遍历训练集trainMatrix中所有文档
		if trainCategory[i]==1:#如果侮辱性词汇出现，则侮辱词汇+1，且文档的总词数+1
			p1Num+=trainMatrix[i]
			p1Denom+=sum(trainMatrix[i])
		else :#如果非侮辱性的词汇出现，则非侮辱的词汇+1，且文档的总词数+1
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
	p1Vect=log(p1Num/p1Denom)#对每个元素做除法求概率
	p0Vect=log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive#返回两个向量和一个概率


#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):#文档矩阵，非侮辱性词汇概率向量，侮辱性词汇概率向量，侮辱性文档概率
	p1=sum(vec2Classify*p1Vec)+log(pClass1)#向量元素相乘后求和再加到类别的对数概率上，等价于概率相乘
	p0=sum(vec2Classify*p0Vec)+log(1-pClass1)
	if p1>p0:
		return 1
	else:
		return 0#否则就是非侮辱性文档

#朴素贝叶斯测试函数
def testingNB():
	listOPosts,listclasses=loadDataSet()#从预先加载中调入数据,调入文档列表以及分类向量
	myVocabList=createVocabList(listOPosts)#构建一个包含所有词的列表
	trainMat=[]#初始化训练数据集
	for postinDoc in listOPosts:#填充训练数据列表
		trainMat.append(setofWords2Vec(myVocabList,postinDoc))#将每篇文档的向量添加到矩阵中
	p0V,p1V,pAb=trainNB0(trainMat,listclasses)#训练文档矩阵，返回概率
	testEntry=['love','my','dalmation']#测试
	thisDoc=array(setofWords2Vec(myVocabList,testEntry))
	print(testEntry,"classified as:",classifyNB(thisDoc,p0V,p1V,pAb))
	testEntry=['my','happy']
	thisDoc=array(setofWords2Vec(myVocabList,testEntry))
	print(testEntry,"classified as:",classifyNB(thisDoc,p0V,p1V,pAb))

#测试算法：使用朴素贝叶斯进行交叉验证
def textParse(bigString):#输入字符串进行切分文本
	import re
	listOfTokens=re.split(r'\W*',bigString)#利用正则表达式来切分句子，分隔符是除单词，数字外的任意字符
	return [tok.lower() for tok in listOfTokens if len(tok)>2]

#测试算法，使用朴素贝叶斯进行交叉验证,贝叶斯垃圾邮件分类器
def spamTest():
	docList=[];classList=[];fullText=[]#初始化数据列表
	for i in range(1,26):#导入文本文件
		wordList=textParse(open('email/spam/%d.txt'%i).read())#切分文本
		docList.append(wordList)#切分的文本以原始列表加入文档列表
		classList.append(1)#更新列表
		wordList=textParse(open('email/ham/%d.txt'%i).read())#切分文本
		docList.append(wordList)
		fullText.extend(wordList)#切分后的文本直接合并成词汇列表
		classList.append(0)#标签列表更新
	vocabList=createVocabList(docList)#创建一个文档包含所有文档中出现的不重复的词的列表
	trainingSet=list(range(50));testSet=[]#初始化训练集和测试集列表
	for i in range(10):#随机构建测试集，随机选取10个样本作为测试样本，并从训练样本中剔除
		randIndex=int(random.uniform(0,len(trainingSet)))#随机得到测试样本的index
		testSet.append(trainingSet[randIndex])#将该样本加入测试集中
		del(trainingSet[randIndex])
	trainMat=[];trainClasses=[]#初始化训练集数据列表和标签列表
	for docIndex in trainingSet:#遍历训练集
		trainMat.append(setofWords2Vec(vocabList,docList[docIndex]))#词表转换到向量，并加入到训练数据列表中
		trainClasses.append(classList[docIndex])#相应的标签也加入到训练标签中
	p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))#朴素贝叶斯分类器训练函数
	errorCount=0#初始化错误计数
	for docIndex in testSet:#遍历测试集进行测试
		wordVector=setofWords2Vec(vocabList,docList[docIndex])#词表转换到向量
		if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:#判断分类标签与原标签是否一致
			errorCount+=1
			print("classification error",docList[docIndex])
	print("thr error rate is:",float(errorCount)/len(testSet))

#######################使用朴素贝叶斯分类器从个人广告中获取区域qingxiang
#RSS源分类器及高频去除函数
def calcMostFreq(vocabList,fullText):#词汇表，全部词汇
	import operator
	freDict={}
	for token in vocabList:
		freDict[token]=fullText.count(token)#统计某个元素出现的次数
	sortedFreq=sorted(freDict.item(),key=operator.itemgetter(1),reverse=True)#对该次数进行排序
	return sortedFreq[:30]#返回最高的30个单词

#RSS源分类器
def localWords(feed1,feed0):#两个RSS源
	import feedparser
	docList=[];classList=[];fullText=[]
	minLen=min(len(feed1['entries']),len(feed0['entries']))#初始化数据列表
	for i in range(minLen):#导入文本文件
		wordList=textParse(feed1['entries'][i]['summary'])#切分文本
		docList.append(wordList)#切分后的文本以原始列表加入文本列表
		fullText.extend(wordList)#切分后的文本直接合并到词汇列表
		classList.append(1)#标签列表更新
		wordList=textParse(feed0['entries'][i]['summary'])
		wordList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList=createVocabList(docList)#获得一个所有文件中不重复词的列表
	top30Words=calcMostFreq(vocabList,fullText)#获得频率最高的30个词汇
	for pairw in top30Words:
		if pairw[0] in vocabList:#第一列为单词
			vocabList.remove(pairw[0])
	trainingSet=range(2*minLen);testSet=[]
	for i in range(20):#随机得到20个样本作为测试样本，并从训练样本中剔除
		randIndex=int(random.uniform(0,len(trainingSet)))#随机得到index
		testSet.append(trainingSet[randIndex])#加入测试集中
		del(trainingSet[randIndex])#从训练集中剔除
	trainMat=[];trainClasses=[]#初始化训练及数据和标签列表
	for docIndex in trainingSet:
		trainMat.append(setofWords2Vec(vocabList,docList[docIndex]))#词表转换到向量，并加入训练集数据列表中
		trainClasses.append(classList[docIndex])#相应的标签也加入训练标签列表中
	p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))#朴素贝叶斯分类器训练函数
	errorCount=0#初始化错误计数
	for docIndex in testSet:#遍历测试集进行测试
		wordVector=setofWords2Vec(vocabList,docList[docIndex])#词表到词向量的转换
		if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
			errorCount+=1
			print("classification error ",docList[docIndex])#并且输出出错文本
	print('the error rate is :',float(errorCount)/len(testSet))#打印错误率
	return vocabList,p0V,p1V

#显示地域性的相关单词,最具表征性词汇显示函数
def getTopWords(ny,sf):
	import operator
	vocabList,p0V,p1V=localWords(ny,sf)#RSS源分类器返回概率
	topNY=[];topSF=[]#初始化列表
	for i in range(len(p0V)):#设定阈值，返回大于阈值的所有词，如果输出信息多，就提高阈值
		if p0V[i]>-4.5:topSF.append((vocabList[i],p0V[i]))
		if p1V[i]>-4.5:topNY.append((vocabList[i],p1V[i]))
	sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
	print("SF**SF**")
	for item in sortedSF:
		print(item[0])
	sortedNY=sorted(topNY,key=lambda paor:pair[1],reverse=True)
	print("NY**NY**NY")
	for item in sortedNY:
		print (item[0])

