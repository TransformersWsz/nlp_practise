#!/usr/bin/env python
# @Time    : 2019/6/20 15:35
# @Author  : Swift  
# @File    : knn_main.py
# @Brief   : implement the knn algorithm with corpus
# @Link    : https://github.com/TransformersWsz/nlp_practise


from my_code.knn.words_polarity import Sentence


class KNN(object):

    def __init__(self):
        pass

    def read_file(self, filename: str, polarity: bool) -> list:
        sentences = []

        with open(filename, "r") as f:
            for line in f:
                pieces = line.split()
                words = {}

                for piece in pieces:
                    words[piece] = words.get(piece, 0) + 1 # 统计这句话的词频
                sentences.append(Sentence(words, polarity))
        return sentences

    def get_sentences(self, positive_filename: str, negative_filename: str) -> tuple:
        positive_sentences = self.read_file(positive_filename, True)
        negative_sentences = self.read_file(negative_filename, False)
        return positive_sentences, negative_sentences

    def cos_between_sentences(self, first_sentence: Sentence, senond_sentence: Sentence) -> float:
        """计算两句话的余弦距离"""
        numerator = sum([first_sentence.words[word] * senond_sentence.words[word] for word in first_sentence.words
                          if word in senond_sentence.words])
        first_sqrt = pow(sum([item * item for item in first_sentence.words.values()]), 0.5)
        second_sqrt = pow(sum([item * item for item in senond_sentence.words.values()]), 0.5)
        denominator = first_sqrt * second_sqrt
        cosine = numerator / denominator
        print("{} <---> {} : {}".format(first_sentence.words, senond_sentence.words, cosine))
        return cosine

    def get_cos_vector_of_test(self, test: list, train: list) -> list:
        """计算每个测试样本与所有训练样本的余弦距离"""
        vector = []
        for test_sentence in test:
            cos_distances = []
            for train_sentence in train:
                cos = self.cos_between_sentences(test_sentence, train_sentence)
                cos_distances.append((cos, train_sentence.polarity))
            cos_distances.sort(key=lambda item: item[0], reverse=True)
            vector.append(cos_distances)
        return vector

    def classify(self, test: list, train: list, k: int) -> float:
        """对测试集进行分类
        :param test: 测试集
        :param train: 训练集
        :param k: 邻居个数
        :return: 准确率
        """
        vector = self.get_cos_vector_of_test(test, train)
        loop = min(k, len(train))
        correct_num = 0    # 正确分类个数
        for (index, test_sentence) in enumerate(test):
            positive_num = 0    # positive邻居个数
            negative_num = 0    # negative邻居个数
            for j in range(loop):
                if vector[index][j][1] == True:
                    positive_num += 1
                else:
                    negative_num += 1

            judge = positive_num >= negative_num
            if test_sentence.polarity == judge:
                correct_num += 1
        return correct_num / len(test)


if __name__ == "__main__":
    knn = KNN()
    positive_filename = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\movie\positive.review"
    negative_filename = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\movie\negative.review"
    sentences = knn.get_sentences(positive_filename, negative_filename)
    train = sentences[0][100:] + sentences[1][100:]
    test = sentences[0][:100] + sentences[1][:100]
    accuracy = knn.classify(test, train, 10)
    print(accuracy)