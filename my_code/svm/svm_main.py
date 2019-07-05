#!/usr/bin/env python
# @Time    : 2019/6/24 16:21
# @Author  : Swift  
# @File    : svm_main.py
# @Brief   : implement the svm algorithm with corpus
# @Link    : https://github.com/TransformersWsz/nlp_practise


from my_code.common.words_polarity import Sentence
from sklearn import svm


class SVM(object):

    def __init__(self):
        pass

    def read_file(self, file_path: str, line_dict: dict, polarity: bool) -> list:
        sentences = []
        with open(file_path, "r") as f:
            for line in f:
                pieces = line.split()
                for piece in pieces:
                    line_dict[piece] += 1    # 统计这句话的词频
                sentences.append(Sentence(line_dict, polarity))
        return sentences

    def get_sentences(self, line_dict: dict, positive_filename: str, negative_filename: str) -> tuple:
        positive_sentences = self.read_file(positive_filename, line_dict, True)
        negative_sentences = self.read_file(negative_filename, line_dict, False)
        return positive_sentences, negative_sentences

    def get_line_dict(self, file_paths: list) -> dict:
        """将每一行句子根据所有单词构造字典"""
        line_dict = {}
        for filename in file_paths:
            with open(filename, "r") as f:
                for line in f:
                    pieces = line.split()
                    for piece in pieces:
                        line_dict[piece] = 0
        return line_dict

    def get_train_and_test(self, sentences: tuple) -> tuple:
        train_X = []
        train_Y = []
        test_X = []
        test_Y = []
        for polarity_sentences in sentences:
            for index, each_object in enumerate(polarity_sentences):
                if index >= 200:
                    train_X.append(list(each_object.words.values()))
                    train_Y.append(int(each_object.polarity))
                else:
                    test_X.append(list(each_object.words.values()))
                    test_Y.append(int(each_object.polarity))
        return (train_X, train_Y), (test_X, test_Y)

    def classify(self, train: tuple, test: tuple) -> float:
        """训练并进行测试，返回准确率"""
        clf = svm.SVC(kernel="rbf", C=0.8, gamma=1)
        clf.fit(train[0], train[1])    # 训练

        acc = clf.score(test[0], test[1])
        return acc


if __name__ == "__main__":
    positive_filepath = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\movie\positive.review"
    negative_filepath = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\movie\negative.review"
    file_paths = [positive_filepath, negative_filepath]
    solution = SVM()
    line_dict = solution.get_line_dict(file_paths)    # 获取字典
    sentences = solution.get_sentences(line_dict, positive_filepath, negative_filepath)
    train, test = solution.get_train_and_test(sentences)
    print(solution.classify(train, test))

    # print(len(train[0][1]), "\n", train[1][1], len(test[0][1]), "\n", test[1][1])
    # print(solution.classify(train, test))