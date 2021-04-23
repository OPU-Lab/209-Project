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
    #f = open('/home/tiandong/GITHUB/opu_ref-master/results/fastyolo/layer_8/'+name1)
    #f = open('tinyyolo/'+name1)
    f = open('/home/tiandong/GITHUB/opu_ref-master/results/yolov3/layer_8/'+name1)
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
                return
    print('PASS')
if __name__ == "__main__":
    #compare_result('dma_output_fm.txt','dma_fm.txt')
    #compare_result('dma_output_ker.txt','dma_ker.txt')
    compare_result('out_adder_a.txt','out_adder_a.txt')
    #compare_result('out_adder_b.txt','out_adder_b.txt')
    #compare_result('out_adder_result.txt','out_adder_res.txt')
    #compare_result('cut_data_16.txt','cut_16.txt')
    #compare_result('cut_data_8.txt','cut_8.txt')
    #compare_result('final_output.txt','out.txt')
    #compare_result('ofm.txt','ofm.txt')
    #compare_result('store_output.txt', 'out.txt')