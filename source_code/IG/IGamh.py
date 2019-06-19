import numpy
import math
import operator

#=========================================================================================
#读取指定路径的文件内容
#=========================================================================================
def readFromFile(path,polarity):
	fp = open(path,'rb')
	alllines = fp.readlines() #一行为一个样本
	docs = []
	for line in alllines:
		line = line.strip().split() #将每行按照空行拆分
		line = [piece for piece in line if piece.isalpha()]
		#line = line.isalpha()
		words = [] #存放一行无重复的词汇表
		doc = []
		for word in line:
			word = word.lower()
			if word not in words:#如果该单词在该行中却不在词汇表中则添加
				words.append(word)
		if len(words) > 0:
			doc.append(words) #一行为一个样本，将其词与极性存放
			doc.append(polarity) 
			docs.append(doc) #将每行样本赋给docs
	return docs

#=========================================================================================
#返回两个文档的词汇
#=========================================================================================
def createDomain(domain):
	pos = readFromFile(r'D:\SUDA\corpus\%s\positive.review' % domain,True)
	neg = readFromFile(r'D:\SUDA\corpus\%s\negative.review' % domain,False)
	return pos,neg

#=========================================================================================
#创建词字典
#=========================================================================================
def creatVocabDict(documents):
	vocabList = [] #存放训练集的词列表
	for doc in documents:
		vocabList.extend(doc[0])
	vocabList = set(vocabList)
	vocabDict = dict([(word,i) for i,word in enumerate(vocabList)]) #便于根据词找到词的对应列数
	vocabDict_word = dict([(i,word) for i,word in enumerate(vocabList)]) #计算得到信息增益后便于根据序号找到对应值
	return vocabDict,vocabDict_word

#=========================================================================================
#将文档根据词表转为矩阵，出现即标为1
#=========================================================================================
def doc2Mat(documents,vocabDict):
	lenOfDocs = len(documents) #文档的长度
	numOfWords = len(vocabDict) #词字典的长度
	docMat = numpy.zeros((lenOfDocs,numOfWords)) #行，列
	for i in range(lenOfDocs):
		for word in documents[i][0]: #第i行的第0列
			if word in vocabDict:
				docMat[i][vocabDict[word]] = 1 #将矩阵的第i行第word单词置1
	return docMat

#=========================================================================================
#创建类别矩阵，记录训练集输入的文档对应类别
#=========================================================================================
def cateVect(documents):
	cateVect = [0] * len(documents)
	for i in range(len(documents)):
		#如果该行的极性为1，矩阵就置为1
		if documents[i][1]: 
			cateVect[i] = 1
		else:
			cateVect[i] = 0
	return numpy.array(cateVect) 

#=========================================================================================
#计算熵,情感类别有两种分类，两种类别的求和
#=========================================================================================
def entropy(cateVect):
	PC1 = numpy.sum(cateVect) / len(cateVect) #求类别PC1的概率
	return PC1,PC1 * numpy.log2(PC1)*(-1) -(1 - PC1) * numpy.log2(1 - PC1)

#=========================================================================================
#计算条件熵
#=========================================================================================
def expectedEntropy(docMat,cateVect):
	PC1 = entropy(cateVect)[0]  #对entropy（）操作后的结果取第一个值
	lenOfDocs = len(docMat) #矩阵长度，行
	wordsNum  = len(docMat[0]) #取0行的长度
	PT = numpy.zeros(wordsNum)
	PT_C1 = numpy.zeros(wordsNum) #统计C1类别出现的文档数
	PT_C0 = numpy.zeros(wordsNum)

	for i in range(lenOfDocs): #从第一行开始统计
		PT += docMat[i] #该词出现的文档数
		if cateVect[i]:
			PT_C1 += docMat[i] #该词出现并且类别为pos的文档数
		else:
			PT_C0 += docMat[i] #该词出现并且类别为neg的文档数

	PTe = PT / lenOfDocs #计算P(t) 用该词出现的文档数 / 文档总数
	PT_C0e = PT_C0 / PT #计算P(Ci|t) 计算t出现为Ci类别概率 
	PT_C1e = PT_C1 / PT #PT_C1e = numpy.ones(wordNum) - PT_C0e

	#排除概率为0的情况,计算log2(P(Ci|t))
	log_PT_C1e = numpy.zeros(wordsNum)
	for i,item in enumerate(PT_C1e):
		if item != 0.0:
			log_PT_C1e[i] = numpy.log2(item)

	log_PT_C0e = numpy.zeros(wordsNum)
	for i,item in enumerate(PT_C0e):
		if item !=0.0:
			log_PT_C0e[i] = numpy.log2(item)

	#计算P(t)H(C|t)=-P(t)[p(c1|t)log2(c1|t)+P(c2|t)log2(c2|t)]
	P1 = - PTe * (PT_C1e * log_PT_C1e + PT_C0e * log_PT_C0e)
	#计算P(-t)                                           
	PTne =  numpy.ones(wordsNum) - PTe
	#计算P(c1|-t)即没有特征t时C1的概率 用不出现t时属于类别C1的文档数 / 不出现t的文档
	PT_C1ne = (numpy.tile(numpy.sum(cateVect),wordsNum) - PT_C1) / (numpy.tile(lenOfDocs,wordsNum) - PT)
	#计算P(C0|-t)
	PT_C0ne = numpy.ones(wordsNum) - PT_C1ne
	#计算log2(C1|-t)
	log_PT_C1ne = numpy.zeros(wordsNum)
	for i,item in enumerate(PT_C1ne):
		if item != 0.0:
			log_PT_C1ne[i] = numpy.log2(item)
	#计算log2（C0|-t）
	log_PT_C0ne = numpy.zeros(wordsNum)
	for i,item in enumerate(PT_C0ne):
		if item != 0.0:
			log_PT_C0ne[i] = numpy.log2(item)

	#计算P(-t)H(C|-t) 用1-P(t)H(c|t)
	P2 = -PTne * (PT_C1ne * log_PT_C1ne + PT_C0ne * log_PT_C0ne)

	return P1 + P2 #返回条件熵

if __name__ == '__main__':
	domain = createDomain('kitchen')
	trains = domain[0][:] + domain[1][:] #训练集
	vocabDict,vocabDict_word = creatVocabDict(trains)
	docMat = doc2Mat(trains,vocabDict)
	cateVect = cateVect(trains)
	#计算IG IG（T）= H(C) - H(C|T)
	gainVect = entropy(cateVect)[1] - expectedEntropy(docMat,cateVect)
	gainList = [(gainVect[i],i,vocabDict_word[i]) for i in range(len(gainVect))] #加上[]表示列表可以修改，否则元组不可更改
	print(len(gainList))
	#利用sorted进行排序，对象是列表，元素是列表的0列，降序
	gainList = sorted(gainList,key = operator.itemgetter(0) ,reverse = True) 
	#只取IG大于1的单词
	gainList1 = []
	for i,item in enumerate(gainList):
		if gainList[i][0] > 0.01:
			gainList1.append(gainList[i]) 
	print(gainList1)
	#print(gainList[:10]) #只取列表的前10





