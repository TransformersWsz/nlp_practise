#!/usr/bin/env python
# @Time    : 2019/7/7 11:06
# @Author  : Swift  
# @File    : naive_bayes_main.py
# @Brief   : implement the naive_bayes algorithm with corpus
# @Link    : https://github.com/TransformersWsz/nlp_practise


import numpy as np


class NaiveBayes(object):

    def __init__(self):
        pass

    def read_file(self, file_path: str, polarity: bool) -> tuple:
        docs = []
        label = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                pieces = [piece for piece in set(line.lower().strip().split()) if piece.isalpha()]
                words = []
                for piece in pieces:
                    words.append(piece)
                docs.append(words)
                label.append(polarity)
        return docs, label

    def get_vocab_dict(self, docs: list) -> dict:
        """构建一个词典，每个单词对应一个索引"""
        vocab = []
        for doc in docs:
            vocab.extend(doc)
        vocab = set(vocab)
        vocab_word_index = dict([(word, i) for i, word in enumerate(vocab)])
        return vocab_word_index

    def word_to_mat(self, vocab_word_index: dict, docs: list) -> list:
        """
        n个单词表示有n个特征，对于每句（每条样本，m个样本）设置其特征值。矩阵大小为 m * n
        :param vocab_word_index: 词典
        :param docs: 样本
        :return: 特征矩阵
        """
        matrix = []
        for doc in docs:
            vec = [0] * len(vocab_word_index)
            for word in doc:
                if word in vocab_word_index:
                    vec[vocab_word_index[word]] = 1
            matrix.append(vec)
        return matrix

    def cal_pro(self, matrix: list, label: list) -> tuple:
        """计算p(x1|1), p(x2|1), ..., p(xn|1) 和 p(x1|0), p(x2|0), ..., p(xn|0)"""
        matrix_length = len(matrix)
        words_num = len(matrix[0])
        pa = sum(label) / matrix_length    # 计算pos的概率
        p0Num = np.ones(words_num)    # 平滑 https://juejin.im/post/5aab706451882555784db5e3
        p1Num = np.ones(words_num)    # 同上

        p0Denom = 2    # 平滑
        p1Denom = 2

        for i in range(matrix_length):
            if label[i] == 0:
                p0Num += matrix[i]
                p0Denom += 1
            else:
                p1Num += matrix[i]
                p1Denom += 1

        p1Vec = np.log(p1Num/p1Denom)
        p0Vec = np.log(p0Num/p0Denom)
        return pa, p0Vec, p1Vec

    def classify(self, test: list, pA: float, p0Vec: np.ndarray, p1Vec: np.ndarray) -> int:
        p0 = sum(test * p0Vec) + np.log(1-pA)
        p1 = sum(test * p1Vec) + np.log(pA)
        return 1 if p1 > p0 else 0


if __name__ == "__main__":
    solution = NaiveBayes()
    positive_filepath = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\electronics\positive.review"
    negative_filepath = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\electronics\negative.review"

    pos, pos_label = solution.read_file(positive_filepath, True)
    neg, neg_label = solution.read_file(negative_filepath, False)

    trains = pos[50:] + neg[50:]
    trains_label = pos_label[50:] + neg_label[50:]
    tests = pos[50:] + neg[50:]
    tests_label = pos_label[50:] + neg_label[50:]

    vocab_word_index = solution.get_vocab_dict(trains)
    train_matrix = solution.word_to_mat(vocab_word_index, trains)
    test_matrix = solution.word_to_mat(vocab_word_index, tests)
    pA, p0Vec, p1Vec = solution.cal_pro(train_matrix, trains_label)
    acc = 0
    for i in range(len(test_matrix)):
        predict = solution.classify(test_matrix[i], pA, p0Vec, p1Vec)
        if predict == tests_label[i]:
            acc += 1
    print("the accuracy is {}".format(acc/len(test_matrix)))
