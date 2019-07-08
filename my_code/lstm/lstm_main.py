#!/usr/bin/env python
# @Time    : 2019/7/6 20:06
# @Author  : Swift  
# @File    : lstm_main.py
# @Brief   : implement the lstm model with corpus
# @Link    : https://github.com/TransformersWsz/nlp_practise


from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation

import numpy as np

from my_code.common.words_polarity import Sentence
from my_code.lstm.setting import *    # 导入LSTM配置


class LSTMClassification(object):

    def __init__(self):
        pass

    def read_file(self, file_path: str, polarity: bool) -> list:
        sentences = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                pieces = [piece for piece in line.strip().split() if piece.isalpha()]
                words = {}

                for piece in pieces:
                    words[piece] = words.get(piece, 0) + 1 # 统计这句话的词频
                sentences.append(Sentence(words, polarity))
        return sentences

    def get_sentences(self, positive_filepath: str, negative_filepath: str) -> tuple:
        positive_sentences = self.read_file(positive_filepath, True)
        negative_sentences = self.read_file(negative_filepath, False)
        return positive_sentences, negative_sentences

    def format_data(self, sentences: tuple) -> list:
        """格式化数据，以便后续的训练、验证和测试"""
        data = []
        for polarity_sentences in sentences:    # pos & neg
            polar = []
            for each_sentence in polarity_sentences:    # each_object in pos or neg
                polar.append([list(each_sentence.words.keys()), each_sentence.polarity])
            data.append(polar)
        return data

    def get_vocab_dict(self, trains: list) -> dict:
        """创建词字典"""
        vocab = []
        for sample in trains:    # pos & neg
            vocab.extend(sample[0])
        vocab = set(vocab)
        vocab_word_index = dict([(word, index+1) for index, word in enumerate(vocab)])
        return vocab_word_index

    def get_word_vec(self, samples: list, vocab_dict: dict) -> tuple:
        """构建词向量"""
        samples_length = len(samples)
        x = []
        y = []
        for i in range(samples_length):
            x.append([vocab_dict[word] for word in samples[i][0] if word in vocab_dict])
            y.append([samples[i][1]])

        x = pad_sequences(x, MAX_LENGTH, padding="post")
        y = pad_sequences(y, MAX_LENGTH, padding="post")
        y = np.expand_dims(y, -1)    # 在最后一列添加一维

        return x, y

    def pipeline(self, vocab_dict: dict, train_x: list, train_y: list, cv_x: list, cv_y: list, test_x: list, test_y: list):
        """
        LSTM 流水线：训练->交叉验证->测试
        :param vocab_dict: 词向量
        :param train_x: 训练集的输入
        :param train_y: 训练集的输出
        :param cv_x: 交叉验证集的输入
        :param cv_y: 交叉验证集的输出
        :param test_x: 测试集的输入
        :param test_y: 测试集的输出
        :return: None
        """
        # 创建模型
        lstm_model = Sequential()
        lstm_model.add(Embedding(input_dim=len(vocab_dict)+1, output_dim=EMBEDDING_DIM, mask_zero=True))
        lstm_model.add(LSTM(output_dim=50, return_sequences=True))
        lstm_model.add(Dropout(0.5))
        lstm_model.add(Dense(1))
        lstm_model.add(Activation("sigmoid"))

        # 编译模型
        lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        # 训练模型
        check_point = ModelCheckpoint(filepath="best_model_cn1.hdf5", verbose=1, save_best_only=True)
        hist = lstm_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=N_EPOCH,
                              validation_data=(cv_x, cv_y), callbacks=[check_point], verbose=2)
        val_loss_list = hist.history["val_loss"]    # 找出错误率最低的那次迭代所用的模型
        best_epoch = val_loss_list.index(min(val_loss_list)) + 1
        lstm_model.load_weights("best_model_cn1.hdf5")

        lstm_model.summary()

        # 用最佳模型对测试集进行测试
        loss, acc = lstm_model.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
        print("Test loss: ", loss)
        print("Test accuracy", acc)


if __name__ == "__main__":
    solution = LSTMClassification()
    positive_filepath = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\kitchen\positive.review"
    negative_filepath = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\kitchen\negative.review"
    sentences = solution.get_sentences(positive_filepath, negative_filepath)

    data_set = solution.format_data(sentences)
    trains = data_set[0][300:] + data_set[1][300:]
    cvs = data_set[0][200:300] + data_set[1][200:300]
    tests = data_set[0][:200] + data_set[0][:200]

    vocab_dict = solution.get_vocab_dict(trains)
    train_x, train_y = solution.get_word_vec(trains, vocab_dict)
    cv_x, cv_y = solution.get_word_vec(cvs, vocab_dict)
    test_x, test_y = solution.get_word_vec(tests, vocab_dict)

    solution.pipeline(vocab_dict, train_x, train_y, cv_x, cv_y, test_x, test_y)
