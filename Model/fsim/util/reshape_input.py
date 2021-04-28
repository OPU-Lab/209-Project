import numpy as np
import math
import argparse

# Command line arguments
parser = argparse.ArgumentParser(description='reshape input ifm')
parser.add_argument('--input',metavar='INPUT', type=str, nargs='?', default='', help='path of input ifm')
parser.add_argument('--layout',metavar='LAYOUT', type=str, nargs='?', default='NCHW', help='transpose input ifm')
parser.add_argument('--padding',metavar='PAD', type=int, nargs='+', default='', help='padding up/down/left/right')
parser.add_argument('--kz',metavar='KZ', type=int, nargs='+', default='', help='kernel size H/W')
parser.add_argument('--fl',metavar='FL', type=int, nargs='?', default='', help='fraction length')


class IfmReshape:
    def __init__(self, args):
        self.input = args.input
        self.padding = args.padding
        self.kz = args.kz
        self.layout = args.layout
        self.fl = args.fl
        self.check()
        
    def check(self):
        assert self.input.endswith(('.npy'))
        assert len(self.padding) == 4
        assert len(self.kz) == 2
        assert self.layout in ['NCHW', 'NHWC']
        assert self.fl != ''
        
    def quantize(self, x, fl):
        base = 2 ** (-fl)
        return np.round(x / base) * base

    def dump_bin(self, x, name, fl):
        assert x.shape[-1] <= 64
        h, w, c = x.shape
        base = 2**fl
        with open(name, 'wb') as f:
            for i in range(h):
                for j in range(w):
                    data = (x[i,j,:] * base).tolist()
                    if c <= 32:
                        tmp = [0 for _ in range(32)] + data + [0 for _ in range(32 - c)]
                    else:
                        tmp = data + [0 for _ in range(64 - c)]
                    f.write(np.array(tmp, dtype=np.int8).tobytes())
                    
    def pad(self, x):
        h, w, c = x.shape
        h_n, w_n = h + self.padding[0] + self.padding[1], w + self.padding[2] + self.padding[3]
        p = np.zeros((h_n, w_n, c))
        p[self.padding[0]:self.padding[0]+h, self.padding[2]:self.padding[2]+w, :] = x
        return p
        
    def run(self):
        ifm = np.load(self.input)
        print('Ifm loaded:', ifm.shape)
        if len(ifm.shape) == 4:
            ifm = ifm[0,...]
        if self.layout == 'NCHW':
            ifm = ifm.transpose(1,2,0)
        ifm = self.quantize(ifm, self.fl)
        ifm = self.pad(ifm)
        h_n, w_n, c = ifm.shape
        c_n = c * self.kz[0] * self.kz[1]
        ifm_reshaped = np.zeros((h_n - self.kz[0] + 1, w_n - self.kz[1] + 1, c_n))
        print('Expected shape after reshape:', ifm_reshaped.shape)
        for i in range(h_n - self.kz[0] + 1):
            for j in range(w_n - self.kz[1] + 1):
                tmp = []
                for k in range(c):
                    tmp = ifm[i:i + self.kz[0], j:j + self.kz[1], k].transpose(1, 0).reshape(np.prod(self.kz), order='F').tolist() + tmp
                ifm_reshaped[i ,j, :] = tmp
                #ifm_reshaped[i, j, :] = ifm[i:i + self.kz[0], j:j + self.kz[1], :].transpose(1, 0, 2).reshape(c_n, order='F')
        self.dump_bin(ifm_reshaped, 'ifm_reshaped.bin', self.fl)
        print('Dumped binary')
        
                
args = parser.parse_args()       
IfmReshape(args).run()
    
