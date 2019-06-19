#coding=utf-8

import codecs #codecs专门用作编码转换
import subprocess #允许你去创建一个新的进程让其执行另外的程序，并与它进行通信，获取标准的输入、标准输出、标准错误以及返回码等。 

#======================================================================
#该函数目的是将分好的输出文件还原成分好的文档类型
#======================================================================

def splitFile(output_path,result_path):
	output = codecs.open(output_path,'r','utf-8')
	result = codecs.open(result_path,'w','utf-8')
	for line in output :
		words = line.strip().split() #将每行按照空格拆分
		if len(words) == 0 :
			result.write('\n') #如果该行长度为0 即输入换行符
		else : #共	CN	S	B
			if words[3] == 'B' or words[3] == 'M' :
				result.write(words[0])
			else :
				result.write(words[0])
				result.write(' ')

#======================================================================
#主函数调用封装好的命令
#======================================================================
if __name__ == '__main__' :
	subprocess.call("crfcmd.bat",shell = True)
	splitFile("output.txt","result.txt")