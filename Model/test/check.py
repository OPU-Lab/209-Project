import numpy as np
import os

def compare_result(name1, name2):
    def hex2int(x):
        bstr = bin(int(x, 16))[2:]
        if len(bstr)<26:
            bstr = bstr[0]*(26-len(bstr))+bstr
        elif len(bstr)>26:
            bstr = bstr[-26:]
        return int(bstr,2)
    f = open(name1)
    lines = f.readlines()
    f.close()
    f = open('./'+name2)
    lines1 = f.readlines()
    f.close()
    for i in range(len(lines1)):
        dat = [hex2int(x) for x in lines[i].strip().split(' ')]
        dat1 = [hex2int(x) for x in lines1[i].strip().split(' ')]
        for j in range(len(dat1)):
            if not dat[j]==dat1[j]:
                print('line',i+1,'(',j+1,')',hex(dat[j]),hex(dat1[j]))
                print(lines[i].strip().split(' ')[j],lines1[i].strip().split(' ')[j])
                raise RuntimeError('Unable to match simulation output with ground truth')
    print('PASS')
if __name__ == "__main__":
    compare_result('gt.txt', 'out.txt')