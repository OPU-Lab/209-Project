'''
This script provides reference input/output for 1st layer of ResNet20 in 8-bit fixed-point
'''
import numpy as np

# wl=8, fl: ifmap=5, ofmap=4, weight=7
ifmap = np.load('conv2d_input_wl8_fl5.npy')
ofmap = np.load('conv2d_output_wl8_fl4.npy')
weight = np.load('weight_wl8_fl7.npy')

# Parameters for the 1st conv2d layer
N, C, H, W = ifmap.shape
Kh, Kw = 3, 3
Sh, Sw = 1, 1
Co = ofmap.shape[1]

def dot(K, L):
    return sum(i[0] * i[1] for i in zip(K, L))

# Simulate how a 16-output PE works    
def PE(ifmap_pe, weight_pe, multiplier = 512, output = 16):    
    pe_output = []
    assert len(ifmap_pe) == multiplier // 2
    assert len(weight_pe) == multiplier
    vector_len = multiplier // output
    for i in range(0, multiplier // 2, vector_len):
        ifmap = ifmap_pe[i : i + vector_len]
        weight = weight_pe[i * 2 : (i + vector_len) * 2]
        w1 = weight[0::2]
        w2 = weight[1::2]
        ip1 = dot(ifmap, w1)
        ip2 = dot(ifmap, w2)
        pe_output += [ip1, ip2]
    assert len(pe_output) == output
    return pe_output

# Dataflow: we transform 3x3 conv2d to 1x1 conv2d by im2col
for i in range(0, H - Kh + 1, Sh):
    for j in range(0, W - Kw + 1, Sw):
        # Data fetch: change ifmap data layout for PE input
        ifmap_flatten = []
        # Flatten each input channel (3,3,3) -> (27)
        for k in range(C):
            ifmap_flatten = ifmap[0, k, i:i + Kh, j:j + Kw].transpose(1, 0).reshape(np.prod(Kh * Kw), order='F').tolist() + ifmap_flatten
        # Alignment (power of 2) : (27) -> (32)
        ifmap_flatten += [0 for _ in range(32 - C * Kh * Kw)]
        # Duplicate * 8 -> (256) to use shared input of two 8-bit multipliers in one DSP
        ifmap_pe = []
        for _ in range(8):
            ifmap_pe += ifmap_flatten
        # Data fetch: change weight data layout for PE input
        weight_pe = []
        tmp = []
        # For each output channel
        for co in range(Co-1, -1, -1):
            weight_flatten = []
            # Flatten each input channel (3,3,3) -> (27) (same as above for ifmap)
            for k in range(C):
                weight_flatten = weight[:, :, k, co].transpose(1, 0).reshape(np.prod(Kh * Kw), order='F').tolist() + weight_flatten
            weight_flatten += [0 for _ in range(32 - C * Kh * Kw)]
            tmp.append(weight_flatten)
        # Interleave two output channel weights for each DSP
        for ii in range(0, Co, 2):
            w1 = tmp[ii]
            w2 = tmp[ii + 1]
            for jj in range(len(w1)):
                weight_pe.append(w1[jj])
                weight_pe.append(w2[jj])
        # PE compute
        #
        # Values for your testbench at each round !!!
        # 
        # Input:
        #   ifmap_pe: 256 8-bit ifmap data of one channel (duplicated ifmap)
        #   weight_pe: 512 8-bit weight data
        # Output:
        #   pe_output: 16 8-bit ofmap data of one channel
        #
        pe_output = PE(ifmap_pe, weight_pe)  
        pe_output = np.array(pe_output)
        reference_output = ofmap[0, :, i, j][::-1]
        assert sum(abs(pe_output - reference_output)) == 0
            