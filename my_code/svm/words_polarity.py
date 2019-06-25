#!/usr/bin/env python
# @Time    : 2019/6/20 15:40
# @Author  : Swift  
# @File    : words_polarity.py
# @Brief   : Sentence Object
# @Link    : https://github.com/TransformersWsz/nlp_practise


class Sentence(object):
    """语句对象

    Attributes:
        _words: dict, 统计这句话中的词频
        _polarity: bool --> True: positive, False: negative. 表示这句话的极性.
    """

    def __init__(self, words, polarity):
        """
        :param words: dict
        :param polarity: bool
        """
        self._words = words
        self._polarity = polarity

    @property
    def words(self):
        return self._words

    @property
    def polarity(self):
        return self._polarity
