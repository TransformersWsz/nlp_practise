#coding = utf-8
from __future__ import division
import codecs
import random
import time
import keras
import numpy as np
seed = 1333
np.random.seed(seed)

from keras.layers import *
from keras.models import *
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils


ISOTIMEFORMAT='%Y-%m-%d %X'
maxlen=1200
nb_classes=2
# max_features=30000
batch_size=32
embedding_dim=100
lstm_output_dim=128
hidden_dim=64


class Document :
	def __init__(self,polarity,words) :
		self.polarity = polarity
		self.words = words


def readFromFile(path,polarity) :
	documents = []
	f = codecs.open(path,'r','utf-8')
	for line in f :
		pieces = line.lower().strip().split()
		words = []
		for piece in pieces :
			words.append(piece)
		documents.append(Document(polarity,words))
	return documents


def createDomain() :
	negative = readFromFile(r'D:\SUDA\corpus\movie\negative.review',False)
	positive = readFromFile(r'D:\SUDA\corpus\movie\positive.review',True)
	return negative,positive


def getLexicon(documents) :
	wordsDict = {}
	for doc in documents :
		for word in doc.words :
			wordsDict[word] = wordsDict.get(word,0) + 1
	wordsDict = sorted(wordsDict.items(),key = lambda x : x[1],reverse = True)
	lexicon = dict([(item[0],i + 1) for i,item in enumerate(wordsDict)])
	return lexicon


def createVec(documents,lexicon) :
	vecs = []
	labels = []
	for doc in documents :
		vec = [lexicon[word] for word in doc.words if word in lexicon]
		vecs.append(vec)
		labels.append(doc.polarity)
	return vecs,labels


def preData(trains,testss,lexicon,max_feature) :
	trainVecs,trainLabels = createVec(trains,lexicon)
	testVecs,testLabels = createVec(tests,lexicon)
	#print(np.shape(trainLabels))

	tokenizer = Tokenizer(num_words = max_feature)
	trainArray = tokenizer.sequences_to_matrix(trainVecs,mode = 'binary')
	testArray = tokenizer.sequences_to_matrix(testVecs,mode = 'binary')
	#print(np.shape(trainArray))

	trainArray=np.reshape(trainArray,(len(trainArray),1,max_feature))
	testArray=np.reshape(testArray,(len(testArray),1,max_feature))
	#print(np.shape(trainArray))

	#trainArray = sequence.pad_sequences(trainVecs, maxlen = maxlen)
	#testArray = sequence.pad_sequences(testVecs, maxlen = maxlen)

	trainLabels = np.array(trainLabels)
	testLabels = np.array(testLabels)
	#print(np.shape(trainLabels))

	return trainArray,testArray,trainLabels,testLabels


def LSTM_model(trainArray,testArray,trainLabels,testLabels,max_feature) :
	model = Sequential()
	model.add(LSTM(512,input_shape=(1,max_feature)))
	#model.add(Embedding(max_feature+1, 100))
	#model.add(LSTM(128))
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	model.fit(trainArray,trainLabels,batch_size=512,epochs=30,verbose=1)

	model.save_weights('train_model.hdf5')
	

	preds=model.predict(testArray,verbose=1)
	predLabel=np.argmax(preds,axis=-1)

	return predLabel


def createPRF(pred_label,real_label,seed):  # 计算prf值 写入文件
	accCount=0
	p=[0]*2
	tp=[0]*2
	fp=[0]*2
	for i in range(len(pred_label)):
		t_label=int(real_label[i])   # 真实类别
		p_label=int(pred_label[i])   # 预测类别

		for index in range(2):  
			if t_label==index:  # t_label 等于当前类别index
				p[index]+=1     # index类别数+1
				if p_label==t_label:  # 预测类别==真实类别
					tp[p_label]+=1   
					accCount+=1
				else:            # 预测类别不等于真实类别
					fp[p_label]+=1
	acc=accCount/len(real_label)   # 正确率

	return acc


if __name__ == '__main__' :
	domain = createDomain()
	#print(list(map(lambda x:len(x),domain)))

	trains = domain[0][100:] + domain[1][100:]
	tests = domain[0][:100] + domain[1][:100]
	#print(len(trains))

	random.seed(1111)
	random.shuffle(trains)
	random.shuffle(tests)

	lexicon = getLexicon(trains + tests)
	#print(len(lexicon))
	max_feature = len(lexicon)

	trainArray,testArray,trainLabels,testLabels = preData(trains,tests,lexicon,max_feature)

	trainLabels2=np_utils.to_categorical(trainLabels,2)
	testLabels2=np_utils.to_categorical(testLabels,2)
	#print(np.shape(trainLabels2))

	predLabel = LSTM_model(trainArray,testArray,trainLabels2,testLabels2,max_feature)

	acc=createPRF(predLabel,testLabels,1111)
	print('acc:',acc)