# coding=utf-8
from __future__ import division
import math
import subprocess

#====================================================================
#Document类 内置变量：极性和词列表
#====================================================================
class Document:
	def __init__(self, polarity, words):
		self.polarity = polarity
		self.words = words
'''
#====================================================================
#去除文本中的空格和非字母元素
#====================================================================
def textParse(line):
	pieces = line.strip().split()
	pieces = [piece for piece in pieces if piece.isalpha()]
	return [piece.lower() for piece in pieces]
'''
#====================================================================
#从指定路径读取文件，以列表形式存储极性和词
#====================================================================
def readFromFile(path, polarity):
	fp = open(path, 'r')
	allLines = fp.readlines()
	docs = []
	for line in allLines:
		pieces=set(line.lower().strip().split())
		pieces=[piece for piece in pieces if piece.isalpha()]
		#line = textParse(line)
		words = []
		for piece in pieces :
			#if word not in words:
			words.append(piece) #这里向量化的结果是存MinghuiAn在即记为1，故先把文本中重复的词去掉
		#if len(words) > 0:
		docs.append(Document(polarity, words))
	fp.close()		
	return docs

#====================================================================
#调用读文件函数，返回文件内容
#====================================================================
def createDomain(domain):
	pos = readFromFile(r'F:\SUDA\corpus\%s\positive.review' % domain, True)
	neg = readFromFile(r'F:\SUDA\corpus\%s\negative.review' % domain, False)
	return pos, neg

#====================================================================
#提取训练集中不重复的词构成词表
#====================================================================
def getLexicon(documents):
	words = []
	for doc in documents:
#		words += doc.words
		words.extend(doc.words)
		# print(type(words))
	words = set(words)
	print(words)
	lexicon = dict([(word, i + 1) for i, word in enumerate(words)])
	print(lexicon)
	print (len(lexicon))
	return lexicon

#====================================================================
#将文本转为svm_light所需要的数据格式 label index：value ……
#====================================================================
def createSvmText(documents, lexicon, outputPath):
	text = ''
	lines = []
	for doc in documents:
		if doc.polarity == True:
			line = "+1 "
		else:
		    line = "-1 "	
		pairs = [(lexicon[word], 1) for word in doc.words if word in lexicon]
		pairs.sort()
		line += ' '.join(['%d:%d' % (pair[0], pair[1]) for pair in pairs])
		text += line + '\n'
	if len(text) > 0:	
		output = open(outputPath, 'w')
		output.write(text)	
		output.close()

#====================================================================
#调用svm_light对应的learn和classify程序完成训练和分类
#====================================================================
def classify(trains, tests):
	lexicon = getLexicon(trains)
	createSvmText(trains, lexicon, 'train.txt')
	createSvmText(tests, lexicon, 'test.txt')
#	subprocess.call('cmd.bat',shell=True)
	proc = subprocess.Popen('cmd.bat', stdout = subprocess.PIPE)
	print (proc.stdout.readlines())
	proc.wait()
	return testing(tests)

#====================================================================
#测试分类的准确性
#====================================================================
def testing(tests):	
	classifiedRes = open('result.output', 'rb')
	accurate = 0
	results = []
	for i, line in enumerate(classifiedRes):
		distance = float(line)
		if((tests[i].polarity == 1 and distance > 0) or (tests[i].polarity == 0 and distance < 0)):
			accurate += 1

		#x0 = 1/(1 + math.exp(abs(distance)))	
		#x1 = 1/(1 + math.exp(-1 * abs(distance))) #x0 + x1 = 1
		#prob = x1/(x0 + x1) #其实就是prob = x1
		#if distance < 0:
			#prob *= -1
		#results.append(prob) #依我拙见，可能就是将distance进行归一化，使得结果的绝对值处于0,1之间，确切作用暂不知

	accuracy = accurate/len(tests)
	print ('The accuracy is:%.2f (%d/%d)' % (accuracy, accurate, len(tests)))
	classifiedRes.close()
	return accuracy, results

if __name__ == '__main__':
	domain = createDomain('movie')
	trains = domain[0][200:] + domain[1][200:]
	tests = domain[0][:200] + domain[1][:200]
	classify(trains, tests)