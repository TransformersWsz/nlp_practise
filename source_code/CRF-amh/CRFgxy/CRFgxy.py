#coding=utf-8
from __future__ import division
import math
import codecs
import subprocess
import os



def train(template_path,train_path,model_path):
    # 通过训练集训练，得到模板文件
    cmd = 'crf_learn.exe %s %s %s' % (template_path, train_path, model_path)
    proc = subprocess.Popen(cmd)
    if proc.wait() < 0:
        print('Training is terminated.')
    # 训练成功
    else:
        print('Training is successful.')

def test(model_path, test_path, output_path):
    cmd = 'crf_test.exe -m %s %s' % (model_path, test_path)
    proc = subprocess.Popen(cmd, stdout=codecs.open(output_path, 'w'))
    if proc.wait() < 0:
        print('Testing is terminated.')
    else:
        print('Testing is successful.')
        cmd = 'perl conlleval.pl < %s' % (output_path)
        os.popen(cmd)


def splitwords(output_path,result_path):
    with codecs.open(result_path,'w','utf-8') as fo:
        with codecs.open(output_path,'r','utf-8') as fi:
            for line in fi:
                words = line.strip().split()
                print(words)
                if len(words)==0:
                    fo.write('\n')
                else:
                    if words[3] == 'B' or words[3] == 'M':
                        fo.write(words[0])
                    else:
                        fo.write(words[0])
                        fo.write(' ')




if __name__ == '__main__':
    # 训练训练集,得到模板文件
    #train('template', 'pku_training.data', 'model')
    # 测试测试集，得到结果对比文件
    #test('model', 'pku_test.data', 'output_crf.txt')
    # 读输出文件，将分词后的文本写入结果文件
    splitwords('output_crf.txt','result.txt')