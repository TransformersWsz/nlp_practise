#!/usr/bin/env python
# @Time    : 2019/6/20 15:54
# @Author  : Swift  
# @File    : test.py
# @Brief   : 
# @Link    :

if __name__ == "__main__":
    filename = r"C:\Users\transformers\Desktop\nlp_practise\corpus\data\movie\negative.review"
    count = 0
    with open(filename, "r") as f:
        for line in f:
            count += 1
            print(line)
        print("total count : {}".format(count))