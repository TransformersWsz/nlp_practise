#-*- coding: utf-8 -*-

#from _future_ import division #导入后执行的精确除法
import math #提供了很多对浮点数和数字的运算函数
import time

'''
求两个特征向量的欧式距离

def euclidea_distance(trains,test):
	words=set(tests.key()+trains.key())#单词列表为测试集和训练集的无重复的元素
	tmp=0
	for word in words:#循环求距离
		tmp+=(tests.get(word,0)-trains.get(word,0))**2 # （测试集中的向量-训练集中的向量）平方求和,若没有单词就返回0
	distances=math.sqrt(tmp)#开根号
	return distance
'''

#求相似度的算法
def cos(source,target):
	numerator = sum(source[word]*target[word] for  word in source if word in target)#向量内积
	sourceLen = math.sqrt(sum([value*value for value in source.values()]))#求原文件向量中的模
	targetLen = math.sqrt(sum([value*value for value in target.values()]))#求目标文件向量中的模

	denominator = sourceLen*targetLen
	if denominator==0:
		return 0
	else :
		return numerator/denominator # 返回cos的值即向量内积/模的乘积

#求相似度矩阵
def similarVector(trains,tests):
	trainLen=len(trains)
	testsLen=len(tests)
	simVector={}#字典中存放测试样本与训练样本的相似矩阵，键值为测试样本的文件序号
	for testId in range(testsLen):
		#distance_list=[] #存放测试样本与所有训练文件样本的欧式距离
		cosineList=[]
		for trainId in range(trainLen):
	#		distance=euclidea_distance(trains[trainId].words,tests[testId].words)#求两个测试与训练样本特征向量的欧式距离
	#		distance_list.append((distance,trainId))
			cosine=cos(trains[trainId].words,tests[testId].words)
			cosineList.append((cosine,trainId))#列表里存放相似度以及训练样本的编号
	#		distance_list.sort()#按照距离远近排序
		cosineList.sort(reverse=True)
	#		simVector[testId]=distance_list #将一个测试样本文件与所有的训练样本文件的欧式距离放到字典里
		simVector[testId]=cosineList
	return simVector

#K近邻算法情感分类

def classify0(trains,tests,k):
	simVector=similarVector(trains,tests)#字典中存放测试样本与训练样本的相似矩阵
	sumAccurate=0 #正确率
	for i in range(len(tests)):
		sumPositive=0 #positive的极性总数
		sumNegative=0 #negative的极性总数
		for  flag in range(k):#取前K个训练样本文件，并统计极性
			index=simVector[i][flag][1]
			if trains[index].polarity:
				sumPositive+=1
			else:
				sumNegative+=1
		if sumPositive>sumNegative:
			if tests[i].polarity:
				sumAccurate+=1
	acc=sumAccurate/len(tests) #计算正确率
	accr=1-acc
	print("\nthe accuracy is :%.2f"%(accr*100)+"%")


class Document(object):
	def __init__(self,polarity,words):
		self.polarity=polarity
		self.words=words

#从指定路径的文件中读取文档信息，并指明文件代表的极性
def readFromFile(path,polarity):
	input=open(path,'rb')
	documents = []#文档列表即文档
	for line in input:
		pieces=line.split()
		pieces=[chenjing for chenjing in pieces if chenjing.isalpha()]
		words={}#文件中的单词对应的词频度，字典
		for piece in pieces:
			word = piece
			word=piece.lower()
			words[word]=words.get(word,0)+1
		if len(words)>0:
			documents.append(Document(polarity,words))#通过词频度和极性构造文档，并存入文档列表做
	return documents

def createDomain():
	pos=readFromFile(r'D:\SUDA\corpus\kitchen\positive.review',True)
	neg=readFromFile(r'D:\SUDA\corpus\kitchen\negative.review',False)
	return pos,neg

if __name__=='__main__':
	domain=createDomain()
	trains=domain[0][50:]+domain[1][50:]#训练样本文档
	tests=domain[0][:50]+domain[1][:50] #测试样本取的范围
	classify0(trains,tests,20)
