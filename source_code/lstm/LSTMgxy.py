from __future__ import division
import numpy as np

np.random.seed(1337)
import tensorflow as tf

tf.set_random_seed(1111)
import codecs
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Sequential


# 一些参数
max_len = 80
batch_size = 32
n_epoch = 25
embedding_dim = 100

class Document(object):
    def __init__(self, polarity, words):
        self.polarity = polarity
        self.words = words


#===================================
# 读文件
#===================================
def readFromFile(filename,polarity):
    # punctuations = [',','.','!','?','-','(',')','"']
    punctuations = '~`!@#$%^&*()-_+={[}]|\\;:\'"<,>.?/'
    v = ' ' * len(punctuations)
    table = str.maketrans(punctuations, v)
    with codecs.open(filename, 'r', 'utf-8') as fp:
        docs = []
        for line in fp:
            line = line.translate(table)
            pieces = set(line.lower().strip().split())
            words = []
            for piece in pieces:
                if piece not in words:
                    words.append(piece)  # 对每一个文档内的词先进行了去重
            if len(words) > 0:
                docs.append(Document(polarity, words))
    return docs


#====================================================
# 建立一个词典，按照词频降序的顺序排序，一个词对应一个编号，编号从1开始
#=====================================================
def getVocabulary(documents):
    vocabSet = set([])  # 集合中的元素不重复
    for doc in documents:
        # 取并集得到不重复的集合
        vocabSet = vocabSet | set(doc.words)
    # 给每个词语一个序号，从1开始
    vocabDict = dict([(word, i+1) for i, word in enumerate(vocabSet)])
    return vocabDict



#=======================================
# 创建向量
#=======================================
def createVec(documents,vocabDict):
    # 将每个词用词字典里的编号来表示
    x = [[vocabDict[word] for word in doc.words if word in vocabDict]for doc in documents]
    y = [[doc.polarity] for doc in documents]

    # right padding
    x = pad_sequences(x, max_len, padding='post')
    y = pad_sequences(y, max_len, padding='post')
    y = np.expand_dims(y, -1)

    return x,y



#===========================================
# 定义网络结构
#===========================================
def train_lstm(vocabDict, train_x, train_y, test_x, test_y):
    print(u'创建模型...')
    model = Sequential()
    model.add(Embedding(input_dim=len(vocabDict)+1,
                        output_dim=embedding_dim,
                        mask_zero=True))

    model.add(LSTM(output_dim=50,
                   return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print(u'编译模型...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(u"训练...")
    model.fit(train_x, train_y, batch_size=batch_size, epochs=n_epoch,
              validation_data=(test_x, test_y))




    print(u"评估...")
    loss, acc = model.evaluate(test_x, test_y, batch_size=batch_size)
    print('Test loss:', loss)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    # 读取正类文本
    pos = readFromFile('1.txt',1)
    # 读取负类文本
    neg = readFromFile('3.txt',0)


    # 正类文本数
    pos_len = len(pos)
    # 负类文本数
    neg_len = len(neg)

    # 构建训练集
    train = pos[int(pos_len*0.2):] + neg[int(neg_len*0.2):]
    # 构建测试集
    test = pos[:int(pos_len*0.2)] + neg[:int(neg_len*0.2)]

    # 构建词字典
    vocabDict = getVocabulary(train + test)
    # 构建train 词向量
    train_x,train_y = createVec(train,vocabDict)
    # 构建test 词向量
    test_x,test_y = createVec(test,vocabDict)

    # 训练模型，预测结果
    train_lstm(vocabDict,train_x,train_y,test_x,test_y)



