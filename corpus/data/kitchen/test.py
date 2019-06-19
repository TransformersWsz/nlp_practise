#coding:utf-8
from __future__ import division
def similarity(file1,file2):
	f1=open(file1,'r')
	line1=f1.readlines()[10]
	word1=[]
	for w in line1.split(' '):
		word1.append(w)
	print(word1)

	f2=open(file2,'r')
	line2=f2.readlines()[10]
	word2=[]
	for w in line2.split(' '):
		word2.append(w)
	print(word2)

	k=0
	for w in word1:
		if(w in word2):
			k+=1
			print(w)
	print(float(k/len(word1)))

if __name__ == '__main__':
	similarity('negative.review','negative1.review')


