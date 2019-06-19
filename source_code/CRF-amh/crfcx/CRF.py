#coding=utf-8
from __future__ import division
import codecs
import subprocess


def splitFile(output_path,result_path) :
	foutput = codecs.open(output_path,'r','utf-8')
	fresult = codecs.open(result_path,'w','utf-8')
	for line in foutput :
		words = line.strip().split()
		if len(words) == 0 :
			fresult.write('\n')
		else :
			if words[3] == 'B' or words[3] == 'M' :
				fresult.write(words[0])
			else :
				fresult.write(words[0])
				fresult.write(' ')


if __name__ == '__main__' :
	#调用命令进行训练和测试
	subprocess.call("cmd.bat",shell = True)
	#展示分词结果
	splitFile("output.txt","result.txt")

