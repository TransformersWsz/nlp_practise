#coding=utf8

from __future__ import division
import os
import codecs
import re
import numpy as np
import math

import gensim
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.utils import np_utils

embedding_dim = 100
max_length = 80
lstm_output_dim = 128
hidden_dim = 64
nb_classes = 7
np.random.seed(1337)

class Document():
	def __init__(self, words, label):
		self.words = words
		self.label = label	


def readFromCNFiles(dir_path):
	label_dict = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'like':4, 'sadness':5, 'surprise':6}
	file_list = os.listdir(dir_path)
	domain = []

	for i in range(len(file_list)):
		doc_list = []
		emotion = file_list[i].split('_')[-1].split('.')[0]

		with codecs.open(os.path.join(dir_path, file_list[i]), 'r', 'utf-8') as fp:
			all_lines = fp.readlines() 
			print (len(all_lines))
	   
			for line in all_lines[:]:
				line = line.strip().split()
				doc_list.append(Document(line, label_dict[emotion]))

		domain.append(doc_list)

	return domain			


def createVectors(documents, word_model):
	x = []
	y = []
	counter = 0

	for doc in documents:
		vectors = [word_model[word] for word in doc.words if word in word_model]

		if len(vectors) < max_length:
			cur_n_words = len(vectors)     
			counter += 1

			for _ in range(max_length - cur_n_words):
				vectors.append(np.zeros(embedding_dim))

		x.append(vectors[:max_length])
		y.append(doc.label)

	print ('counter:%d' % counter)	

	return x, y


def figures(history, figure_name = "plots"):  
    """ 
    method to visualize accuracies and loss vs epoch for training as well as testind data\n 
    Argumets: history = an instance returned by model.fit method
              figure_name = a string representing file name to plots. By default it is set to "plots"  
    Usage: hist = model.fit(X,y)             
    figures(hist) 
    """  
    from keras.callbacks import History  

    if isinstance(history, History):  
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  
        hist = history.history   
        epoch = history.epoch  
        loss = hist['loss']  
        val_loss = hist['val_loss']  
        plt.figure(1)  
         
        plt.subplot(222)  
        plt.plot(epoch, loss)  
        plt.title("Training loss vs Epoch")  
        plt.xlabel("Epoch")  
        plt.ylabel("Loss")      
  
        plt.subplot(224)  
        plt.plot(epoch, val_loss)  
        plt.title("Validation loss vs Epoch")  
        plt.xlabel("Epoch")  
        plt.ylabel("Validation Loss")    
        plt.tight_layout()  
        plt.savefig(figure_name)  
    else:  
        print ("Input Argument is not an instance of class History")	


def lstmModel(X_train, X_val, X_test, Y_train, Y_val, Y_test):
	input = Input(shape = (max_length, embedding_dim))
	lstm = LSTM(128, dropout_W = 0.5, dropout_U = 0.5)(input)
	dense = Dense(64, activation = 'relu')(lstm)
	do_dense = Dropout(0.5)(dense)
	classification = Dense(7, activation = 'softmax')(do_dense)

	model = Model(input = [input], output = [classification])

	optmr = optimizers.Adadelta()
	# optmr = optimizers.Adagrad(lr = 0.01, epsilon = 1e-06)
	model.compile(loss = 'categorical_crossentropy', optimizer = optmr, metrics = ['accuracy'])
	print ('Model construction complete.')

	check_pointer = ModelCheckpoint(filepath = 'best_model_cn1.hdf5', verbose = 1, save_best_only = True)
	hist = model.fit(X_train, Y_train, batch_size = 32, nb_epoch = 80, verbose = 2, validation_data = (X_val, Y_val), callbacks = [check_pointer])

	val_loss_list = hist.history['val_loss']
	best_epoch = val_loss_list.index(min(val_loss_list)) + 1

	model.load_weights('best_model_cn1.hdf5')
	results = model.predict(X_test, verbose = 1)
	print (results.shape)
	Y_predict = [list(item).index(max(item)) for item in results]

	with open('cn_p1.txt', 'w') as fp:
		for i in range(len(results)):
			fp.write(str(Y_test[i]) + '\t')
			for p in results[i]:
				fp.write(str(p) + '\t')
			fp.write('\n')	

	# accuracy = np_utils.accuracy(Y_predict, Y_test)

	print ('Test Number:%d' % len(Y_test))
	# print ('Test Accuracy:%.6f' % accuracy)

	figures(hist, figure_name = "plots_cn1")
	calFScore(Y_predict, Y_test)


def calFScore(Y_predict, Y_test):
	true_num = [0] * nb_classes
	pred_num = [0] * nb_classes
	act_num = [0] * nb_classes

	for pred_l, act_l in zip(Y_predict, Y_test):
		pred_num[pred_l] += 1
		act_num[act_l] += 1
		
		if pred_l == act_l:
			true_num[pred_l] += 1	

	precision = [0] * nb_classes
	recall = [0] * nb_classes
	f1 = [0] * nb_classes

	for i in range(nb_classes):
		if pred_num[i]:
			precision[i] = true_num[i] / pred_num[i]

		recall[i] = true_num[i] / act_num[i]

		if (precision[i] + recall[i]):
			f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])

	accuracy = sum(true_num) / len(Y_test)	

	for i in range(nb_classes):
		print ('P:%.5f  ' % precision[i] + 'R:%.5f  ' % recall[i] + 'F:%.5f  ' % f1[i])

	print ('Aver_P:%.5f' % (sum(precision) / nb_classes))
	print ('Aver_R:%.5f' % (sum(recall) / nb_classes))
	print ('Aver_F:%.5f' % (sum(f1) / nb_classes))
	print ('Accuracy:%.5f' % accuracy)

	with open('result1.txt', 'a') as fp:
		fp.write('Aver_P:%.5f' % (sum(precision) / nb_classes) + '\t')
		fp.write('Aver_R:%.5f' % (sum(recall) / nb_classes) + '\t')
		fp.write('Aver_F:%.5f' % (sum(f1) / nb_classes) + '\t')
		fp.write('Accuracy:%.5f' % accuracy + '\n')	


def test(trains, vals, tests, word_model):
	train_vecs, train_labels = createVectors(trains, word_model)
	val_vecs, val_labels = createVectors(vals, word_model)
	test_vecs, test_labels = createVectors(tests, word_model)

	X_train = np.reshape(train_vecs, (len(train_vecs), max_length, embedding_dim))
	X_val = np.reshape(val_vecs, (len(val_vecs), max_length, embedding_dim))
	X_test = np.reshape(test_vecs, (len(test_vecs), max_length, embedding_dim))

	print (X_train.shape)
	print (X_val.shape)
	print (X_test.shape)

	Y_train = np_utils.to_categorical(train_labels, nb_classes)
	Y_val = np_utils.to_categorical(val_labels, nb_classes)
	print (Y_train.shape)
	print (Y_val.shape)

	lstmModel(X_train, X_val, X_test, Y_train, Y_val, test_labels)


if __name__ == '__main__':
	dir_path = r'/data/lzhang/Corpus/raw_corpus1_split'
	domain = readFromCNFiles(dir_path)

	word_model = gensim.models.Word2Vec.load('word_model_cn_100.m')
	print (word_model)

	trains, vals, tests = [], [], []
	for i in range(nb_classes):
		print (len(domain[i]))
		if i == 2:
			for _ in range(6):
				trains += domain[i][108:]
			trains += domain[i][108:122]
		else:		
			trains += domain[i][108:362]

		vals += domain[i][72:108]
		tests += domain[i][:72]	

	np.random.shuffle(trains)
	print (len(trains), len(vals), len(tests))	

	test(trains, vals, tests, word_model)