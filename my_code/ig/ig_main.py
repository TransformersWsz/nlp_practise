#!/usr/bin/env python
# @Time    : 2019/7/5 9:55
# @Author  : Swift  
# @File    : ig_main.py
# @Brief   : implement the ig algorithm with corpus
# @Link    : https://github.com/TransformersWsz/nlp_practise


import numpy as np
from my_code.common.words_polarity import Sentence

class IG(object):
    """
    IG算法的具体讲解见：
        https://blog.csdn.net/aws3217150/article/details/49906389
        https://blog.csdn.net/It_BeeCoder/article/details/79554388
    """

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

    def get_trains(self, sentences: tuple) -> list:
        trains = []
        for polarity_sentences in sentences:    # pos & neg
            for each_sentence in polarity_sentences:    # each_object in pos or neg
                trains.append([list(each_sentence.words.keys()), each_sentence.polarity])
        return trains

    def get_vocab_dict(self, trains: list) -> tuple:
        """创建词字典"""
        vocab = []
        for sample in trains:    # pos & neg
            vocab.extend(sample[0])
        vocab = set(vocab)
        vocab_word_index = dict([(word, index) for index, word in enumerate(vocab)])
        vocab_index_word = dict([(index, word) for index, word in enumerate(vocab)])
        return vocab_word_index, vocab_index_word

    def get_word_matrix(self, trains: list, vocab_word_index: dict) -> np.ndarray:
        """创建矩阵，在每个样本中单词出现即设置为1，未出现即设置为0"""
        trains_length = len(trains)
        vocab_length = len(vocab_word_index)
        word_matrix = np.zeros((trains_length, vocab_length))

        for i in range(trains_length):
            for word in trains[i][0]:
                if word in vocab_word_index:
                    word_matrix[i][vocab_word_index[word]] = 1
        return word_matrix

    def get_polarity_matrix(self, trains: list) -> np.ndarray:
        """创建类别矩阵"""
        polarity_matrix = np.zeros(len(trains))

        for i, item in enumerate(trains):
            if item[1] == True:
                polarity_matrix[i] = 1
        return polarity_matrix

    def get_total_entropy(self, polarity_matrix: np.array) -> tuple:
        """计算整体熵"""
        pc1 = np.sum(polarity_matrix) / len(polarity_matrix)    # pc1: postive
        return pc1, -1*pc1*np.log2(pc1) - (1-pc1)*np.log2(1-pc1)

    def get_conditional_entropy(self, word_matrix: np.ndarray, polarity_matrix: np.ndarray) -> np.ndarray:
        """计算条件熵"""
        trains_length = len(word_matrix)
        word_length = len(word_matrix[0])
        PT = np.zeros(word_length)
        PT_C1 = np.zeros(word_length)
        PT_C0 = np.zeros(word_length)

        for i in range(trains_length):
            PT += word_matrix[i]
            if polarity_matrix[i] == True:
                PT_C1 += word_matrix[i]    # 该词出现并且类别为pos的文档数
            else:
                PT_C0 += word_matrix[i]    # 该词出现并且类别为neg的文档数

        PTe = PT / trains_length  # 计算P(t) 用该词出现的文档数 / 文档总数
        PT_C0e = PT_C0 / PT  # 计算P(Ci|t) 计算该词出现为neg概率
        PT_C1e = PT_C1 / PT  # 计算P(Ci|t) 计算该词出现为pos概率

        log_PT_C1e = np.zeros(word_length)    # 排除概率为0的情况,计算log2(P(Ci|t))
        for i, item in enumerate(PT_C1e):
            if item != 0.0:
                log_PT_C1e[i] = np.log2(item)

        log_PT_C0e = np.zeros(word_length)
        for i, item in enumerate(PT_C0e):
            if item != 0.0:
                log_PT_C0e[i] = np.log2(item)

        P1 = - PTe * (PT_C1e * log_PT_C1e + PT_C0e * log_PT_C0e)    # 计算该词出现 类别为neg何pos的概率

        PTne = np.ones(word_length) - PTe    # 计算该词没有出现的概率
        PT_C1ne = (np.tile(np.sum(polarity_matrix), word_length) - PT_C1) / \
                  (np.tile(trains_length, word_length) - PT)    # 类别为pos且该词没有出现的个数 | 该词没有出现的个数 ===》 该词没有出现情况下类别为pos的概率

        PT_C0ne = np.ones(word_length) - PT_C1ne
        log_PT_C1ne = np.zeros(word_length)
        for i, item in enumerate(PT_C1ne):
            if item != 0.0:
                log_PT_C1ne[i] = np.log2(item)
        # 计算log2（C0|-t）
        log_PT_C0ne = np.zeros(word_length)
        for i, item in enumerate(PT_C0ne):
            if item != 0.0:
                log_PT_C0ne[i] = np.log2(item)

        P2 = - PTne * (PT_C1ne * log_PT_C1ne + PT_C0ne * log_PT_C0ne)

        return P1 + P2




if __name__ == "__main__":
    solution = IG()
    positive_filepath = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\kitchen\positive.review"
    negative_filepath = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\kitchen\negative.review"
    sentences = solution.get_sentences(positive_filepath, negative_filepath)
    trains = solution.get_trains(sentences)

    vocab_word_index, vocab_index_word = solution.get_vocab_dict(trains)
    word_matrix = solution.get_word_matrix(trains, vocab_word_index)
    polarity_matrix = solution.get_polarity_matrix(trains)

    information_gain = solution.get_total_entropy(polarity_matrix)[1] - \
                       solution.get_conditional_entropy(word_matrix, polarity_matrix)    # 计算得IG

    ig_list = [(information_gain[i], i, vocab_index_word[i]) for i in range(len(information_gain))]
    ig_list.sort(key=lambda item: item[0], reverse=True)

    for item in ig_list:
        print(item)
    print(ig_list[:10])


