#!/usr/bin/env python
# @Time    : 2019/7/3 17:31
# @Author  : Swift  
# @File    : crf_main.py
# @Brief   : implement the crf algorithm with CRF++
# @Link    : https://github.com/TransformersWsz/nlp_practise


import subprocess


class CRF(object):

    def __init__(self):
        pass

    def call_sys_shell(self, shell_path: str, is_shell: bool):
        """调用系统命令执行bat"""
        subprocess.call(shell_path, shell=is_shell)

    def restore_file(self, segmented_file: str, output_file: str, separator: str):
        """
        将分好词的文件还原成一片文章
        :param segmented_file: 分好词的文件
        :param output_file: 输出文件
        :param separator: 分割符
        :return: None
        """
        with open(output_file, "w", encoding="utf-8") as fw:
            with open(segmented_file, "r", encoding="utf-8") as fr:
                for line in fr:
                    words = line.strip().split()
                    if len(words) == 0:
                        fw.write("\n")
                    else:
                        if words[3] == "B" or words[3] == "M":
                            fw.write(words[0])
                        else:
                            fw.write(words[0] + separator)

if __name__ == "__main__":
    solution = CRF()
    solution.call_sys_shell("crfcmd.bat", True)
    solution.restore_file("output.txt", "article.txt", " ")