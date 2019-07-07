import numpy as np 
import math
import operator
np.random.seed(1337) #方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数
import tensorflow as tf 
tf.set_random_seed(1111) #设置图级随机seed



from keras.models import Model
from keras.preprocessing.sequence import pad_sequences #用于允许快速的创建能处理序列信号的模型，可以很快的将一个图像分类的模型变为一个对视频分类的模型
from keras.layers import * #使用lstm做序列分类
from keras.models import Sequential #通过将层的列表传递个哦Sequential构造函数，创建一个Sequen模型
from keras.callbacks import ModelCheckpoint
from keras import optimizers


#参数设置
max_len = 80 #用于设置序列填充的最大值
batch_size = 32 #keras训练模型中用于每经多少个样本就更新一次权重，默认32
n_epoch = 25 #训练的轮数
embedding_dim = 100 #用于嵌入层的全连接嵌入的维度


#=================================================================================================
#读取指定路径的文件
#=================================================================================================
def readFromFile(path,polarity):
	fp = open(path,'r',encoding = 'utf-8')
	alllines = fp.readlines()
	docs = [] #存放整个训练集样本的词汇
	for line in alllines:
		line = line.strip().split()
		line = [piece for piece in line if piece.isalpha()]
		words = []
		doc = []
		for word in line:
			word = word.lower()
			if word not in words:
				words.append(word)
		if len(words) > 0:
			doc.append(words)
			doc.append(polarity)
			docs.append(doc)
	return docs

#=========================================================================================
#返回两个文档的词汇
#=========================================================================================
def createDomain(domain):
	pos = readFromFile(r'F:\SUDA\corpus\%s\positive.review' % domain,True)
	neg = readFromFile(r'F:\SUDA\corpus\%s\negative.review' % domain,False)
	return pos,neg

#=========================================================================================
#建立一个词典，按照词频降序的顺序排序，一个词对应一个编号，从1开始编号
#=========================================================================================
def createVocabDict(documents):
	vocabList = []
	for doc in documents:
		vocabList.extend(doc[0])
	vocabList = set(vocabList)
	vocabDict = dict([(word,i+1) for i ,word in enumerate(vocabList)])
	return vocabDict

# def createVocabDict(documents):
#     vocabSet = set([])  # 集合中的元素不重复
#     for doc in documents:
#         # 取并集得到不重复的集合
#         vocabSet = vocabSet | set(doc[0])
#     # 给每个词语一个序号，从1开始
#     vocabDict = dict([(word, i+1) for i, word in enumerate(vocabSet)])
#     return vocabDict

#=========================================================================================
#建立词向量
#=========================================================================================
def createVec(documents,vocabDict):
	lenOfDocs = len(documents)
	print (lenOfDocs)
	# x = []
	# y = []
	# for i in range(lenOfDocs):
	# 	x.append([vocabDict[word] for word in documents[i][0] if word in vocabDict])
	# 	y.append([documents[i][1]])
	x = [[vocabDict[word] for word in documents[i][0] if word in vocabDict] for i in range(lenOfDocs)]
	y = [[documents[j][1]] for j in range(lenOfDocs)]

	x = pad_sequences(x,max_len,padding = 'post')
	y = pad_sequences(y,max_len,padding = 'post')
	y = np.expand_dims(y,-1) #在最后一列添加一维

	return x,y


#=========================================================================================
#训练过程
#=========================================================================================
def train_lstm(vocabDict,train_x,train_y,val_x, val_y, test_x,test_y):
	print("第一步创建模型")
	model = Sequential() #通过.add()方法一个个将layer加入模型中
	model.add(Embedding(input_dim = len(vocabDict) + 1,output_dim = embedding_dim,mask_zero = True))
	model.add(LSTM(output_dim = 50,return_sequences = True)) # false
	model.add(Dropout(0.5)) #防止过拟合，过拟合就是训练过程很优越，而测试过程会欠佳
	model.add(Dense(1)) #标准的一维全连接层
	model.add(Activation('sigmoid')) #激活函数

	print("第二步编译模型")
	model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

	print("第三步训练模型")
	check_pointer = ModelCheckpoint(filepath = 'best_model_cn1.hdf5',verbose = 1, save_best_only = True)
	hist = model.fit(train_x,train_y,batch_size = batch_size,epochs = n_epoch,validation_data = (val_x,val_y), callbacks = [check_pointer],verbose = 2)
	val_loss_list = hist.history['val_loss'] #找出错误率最低的那次迭代所用的模型
	best_epoch = val_loss_list.index(min(val_loss_list)) + 1
	model.load_weights('best_model_cn1.hdf5')


	model.summary()


	print ("第四步评估结果")
	loss,acc = model.evaluate(test_x,test_y,batch_size = batch_size)
	print ('Test loss:' ,loss)
	print ('Test accuracy:',acc)



if __name__ == '__main__':
	domain = createDomain('kitchen')
	trains = domain[0][300:] + domain[1][300:] #训练集
	print(trains)
	vals =  domain[0][200:300] + domain[1][200:300]
	tests = domain[0][:200] + domain[1][:200] #测试集
	print (len(trains), len(tests))
	vocabDict = createVocabDict(trains)
	train_x,train_y = createVec(trains,vocabDict)
	val_x,val_y = createVec(vals,vocabDict)
	test_x,test_y = createVec(tests,vocabDict)
	print (len(train_x), len(train_y))
	print (len(test_x), len(test_y))
	train_lstm(vocabDict,train_x,train_y, val_x, val_y, test_x,test_y)



	
#=================================================================================================
# 将序列进行填充
# keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',
#   padding='pre', truncating='pre', value=0.)
# maxlen:填充的序列的最大长度
# dtype:返货的numpy array的数据类型
# padding：pre或者post进行填充
# truncating:用于对序列进行截断从起始还是结尾截断
# value:浮点数，此值将在填充时代替默认的填充值0
#==================================================================================================

#===============================================================================================
# 在训练模型之前，需要对学习过程进行配置
# model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
# optimizer：指定为已预定义的优化器名
# loss：损失函数，该参数为模型试图最小化的目标函数
# metrics：指标列表，对于分类问题，我们一般将该列表设置为metrics="accuracy"
#===============================================================================================
#===============================================================================================
# 训练模型
# model.fit(X_train, y_train, epochs=50, batch_size=128)
# x,y为训练数据
# epochs:训练的轮数
# batch_size：每经过多少个sample更新一次权重
# validation_data=None:验证集
#===============================================================================================
#=================================================================================================
# keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform',
#    embeddings_regularizer=None, activity_regularizer=None, 
#    embeddings_constraint=None, mask_zero=False, input_length=None)
# 嵌入层将正整数下标转化为具有固定大小的向量，作为模型的第一层
# input_dim：字典长度，即输入数据最大下标+1
# output_dim：代表全连接嵌入的维度
# embeddings_initialier：嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器
# embeddings_regularizer：嵌入矩阵的正则项
# embedding_constraint：嵌入矩阵的约束项
# mask_zero：布尔值，确定是否将输入的0看作是应该被忽略的填充，该参数在递归层处理边长输入时有用，如果是True，则模型的后续层
# 都要支持masking，并且下标0在字典中不可用
# input_length:输入序列的长度固定时，则该值为其长度，若要接Flatten Dense层就要指定参数，否则Dense层的输出维度无法自动推断
#=================================================================================================