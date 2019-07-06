#!/usr/bin/env python
# @Time    : 2019/7/6 20:09
# @Author  : Swift  
# @File    : setting.py
# @Brief   : implement the lstm model with corpus
# @Link    : https://github.com/TransformersWsz/nlp_practise


# LSTM 参数设置

MAX_LENGTH = 80    # 设置序列填充的最大值
BATCH_SIZE = 32    # 每个epoch训练32个样本
N_EPOCH = 25    # 训练轮数
EMBEDDING_DIM = 100    # 嵌入层的全连接嵌入的维度