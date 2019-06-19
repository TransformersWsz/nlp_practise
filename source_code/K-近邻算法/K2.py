
from numpy import *
import operator
from os import listdir
#将图像格式化处理为一个向量 32*32图像转换为1*1024向量
"""
首先创建1*1024的numpy数组
然后打开给定的文件，读出文件前32行，
将每行的头32个字符存储在numpy数组中
"""
def img2vector(filename):
	returnVect=zero((1,1024))#初始化要返回1*1024向量
	fr=open(filename)
	for i in range(32):#循环读取文件的前32行
		lineStr=fr.readline()
		for j in range(32):#每行的头32个字符存储到要返回的向量中
			returnVect[0,32*i+j]=int(lineStr[j])
	return returnVect#返回输出的1*1024向量
