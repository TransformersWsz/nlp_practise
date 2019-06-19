# coding=utf-8
from __future__ import division
import math
from numpy import *
import codecs

#==================================================================
# 读文件
#==================================================================
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


#==================================================================
#创建一个包含所有文档中出现的不重复词及其序号的字典
#==================================================================
def createVocabList(documents):
	vocabSet = set([]) #集合中的元素不重复
	for doc in documents:
        # 取并集得到不重复的集合
		vocabSet = vocabSet | set(doc)
    # 给每个词语一个序号，从0开始
	vocabDict = dict([(word, i) for i, word in enumerate(vocabSet)])
	return vocabDict

#==================================================================
#输入是词汇表和某文档，输出是文档向量
#向量的每一个元素为0或1，表示词汇表中的单词是否出现在输入文档中
#==================================================================
def setOfWords2Vec(vocabDict, inputSet):
	martix = []
    # 对每一个句子
	for inputLine in inputSet:
        # 向量长度为词总数
		returnVec = [0] * len(vocabDict)
        # 对句子中的每一个词
		for word in inputLine:
            # 若该词在词典中
			if word in vocabDict:
                # 词对应序号位置置1
				returnVec[vocabDict[word]] = 1
        #将此行加入到矩阵之中
		martix.append(returnVec)
	return martix

#=================================================================
#计算每个类别的概率，以及类别下每个特征属性划分的条件概率
#=================================================================
def trainNB0(trainMartix, trainCategory):
    # 训练集句子总数
	numTrainDocs = len(trainMartix)
    # 训练集中词语总数
	numOfWords = len(trainMartix[0])
	pAbusive = sum(trainCategory) / numTrainDocs #因类别标记为0和1，故sum(trainCategory)即为类别为1的文档数
    # 初始化类别为0的向量
	p0Num = ones(numOfWords)
    # 初始化类别为1的向量
	p1Num = ones(numOfWords) #Laplace校准，避免概率为0的情况
    #
	p0Denom = 2
	p1Denom = 2
    # 对每一条句子
	for i in range(numTrainDocs):
        # 如果该条句子的类别属于0
		if trainCategory[i] == 0:
            # 将该行叠加到p0Num中，计算所有类别为0的句子中每个词出现的次数
			p0Num += trainMartix[i] #向量加法
            # 计算所有类别为0的句子的词语总数
			p0Denom += sum(trainMartix[i]) #整行元素相加
        # 如果该条句子的类别属于1
		else:
            # 将该行叠加到p1Num中，计算所有类别为1的句子中每个词出现的次数
		    p1Num += trainMartix[i]
            # 计算所有类别为1的句子的词语总数
		    p1Denom += sum(trainMartix[i])
    # 向量除法 得到在类别0的情况下，每个词的词频:p(w1|p0),p(w2|p0),...,p(wn|p0)
	p0Vec = log(p0Num / p0Denom)
    # 向量除法 得到在类别1的情况下，每个词的词频:p(w1|p1),p(w2|p1),...,p(wn|p1)
	p1Vec = log(p1Num / p1Denom) #避免下溢出，采用log
	return pAbusive, p0Vec, p1Vec

#=================================================================
#计算测试文档对应的概率，确定测试文档的类别
#=================================================================
def classifyNB(test, pAbusive, p0Vec, p1Vec):
    # p(y=0|w1,w2,w3,...,wn)
    # sum中计算每句话中每个词的概率的和
	p0 = sum(test * p0Vec) + log(1 - pAbusive)
    # p(y=1|w1,w2,w3,...,wn)
	p1 = sum(test * p1Vec) + log(pAbusive)
    # 由于p在0-1之间，大部分会特别小，python相乘这类数会四舍五入为0，所以取log（递增），不影响答案
	if p0 > p1:
		return 0
	else:
	    return 1

#=================================================================
#将测试结果与真实标签进行比较，计算准确率
#=================================================================
def testingNB():
    # 读文件，获得正面的数据集和正面标签
	pos, posLabel = readFromFile(r'D:\SUDA\corpus\kitchen\positive.review',1)
    # 读文件，获得负面的数据集和负面标签
	neg, negLabel = readFromFile(r'D:\SUDA\corpus\kitchen\negative.review',0)

    # 正面句子总数
	posTotal = len(posLabel)
    # 负面句子总数
	negTotal = len(negLabel)
    #组成训练集和测试集
	trains = pos[int(posTotal*0.2):] + neg[int(negTotal*0.2):] #选取pos与neg的后百分之八十作为训练集
	tests = pos[:int(posTotal*0.2)] + neg[:int(negTotal*0.2)] #选取pos与neg的前百分之二十作为测试集
	trainCategory = posLabel[int(posTotal*0.2):] + negLabel[int(negTotal*0.2):]
	testCategory = posLabel[:int(posTotal*0.2)] + negLabel[:int(negTotal*0.2)]

    #创建词典，每个词对应一个序号
	vocabDict = createVocabList(trains)
    #构建训练集的矩阵，每一行句子用一个行向量表示，向量维度为训练集的词总数，句子中出现一个词就在此相应的序号位置置1
	trainMartix = setOfWords2Vec(vocabDict, trains)
    # 构建测试集的矩阵
	testMartix = setOfWords2Vec(vocabDict, tests)

    #获得测试集的句子总数
	testLen = len(testMartix)
    # 利用朴素贝叶斯计算：p(类别为1的句子数/总句子数),[p(w1|p0),p(w2|p0),...,p(wn|p0)],[p(w1|p1),p(w2|p1),...,p(wn|p1)]
	pAbusive, p0Vec, p1Vec = trainNB0(trainMartix, trainCategory)
    # 准确率
	accurate = 0
    # 对每个测试集的句子
	for k in range(testLen):
        # 将上面得到的概率和测试集矩阵代入计算，p(p0|w0,w1,w2....wn)和p(p1|w0,w1,w2....wn)
		predictLabel = classifyNB(testMartix[k], pAbusive, p0Vec, p1Vec)
        # 若预测正确
		if predictLabel == testCategory[k]:
			accurate += 1
	print("The accuracy is:%.2f" % (accurate / testLen * 100) + "%")

if __name__ == '__main__':
	testingNB()