import json
import numpy as np
import argparse
import os
import torch 
import torch.nn as nn

# Command line arguments
parser = argparse.ArgumentParser(description='generate golden results')
parser.add_argument('--config',metavar='IR', type=str, nargs='?', default='OPU_IR.json', help='path of input json')
parser.add_argument('--input',metavar='INPUT', type=str, nargs='?', default='', help='path of input fm')
parser.add_argument('--weight_dir',metavar='WDIR', type=str, nargs='?', default='dump', help='path of dumped weights')

class GoldenGen:
    def __init__(self, args):
        self.config = args.config
        self.input = args.input
        self.wdir = args.weight_dir
        self.quantize_fm = True
        self.fm_dw = 8
        self.var_dict = {}
        self.check()
        
    def check(self):
        dat = np.load(self.input)
        self.var_dict[0] = torch.from_numpy(dat)
        self.ir = {}
        with open(self.config) as json_file: 
            lines = json_file.readlines()
            for line in lines:
                ir = json.loads(line)
                self.ir[int(ir['index'])] = ir
        
    def pooling(self, x, ir):
        if ir['pooling_type'] == 1:
            kernel_size = ir['pooling_size'][1:3]
            stride = ir['pooling_stride'][1:3]
            maxpool2d = nn.MaxPool2d(kernel_size, stride)
            print(maxpool2d)
            x = maxpool2d(x)
        return x
            
    def activation(self, x, ir):
        act_opcode = ir['activation_type']
        if act_opcode == 1:
            relu = nn.ReLU()
            print(relu)
            x = relu(x)
        elif act_opcode > 0:
            assert 0
        return x
    
    def quantize(self, x, ir, info):
        if self.quantize_fm:
            if info == 'input':
                fl = ir['input_fraclen']
            elif info == 'output':
                fl = ir['output_fraclen']
            else:
                assert 0
            base = 2 ** (-fl)
            dat = x.detach().numpy()
            dat = np.maximum(-2**self.fm_dw, np.minimum(2**self.fm_dw - 1, np.floor(dat / base))) * base
            x = torch.from_numpy(dat)
            print('quantize', info, ' fraclen:', fl)
        return x    
        
    def run_layer(self, ir, params):        
        if len(params['input']) == 1:
            x = params['input'][0]
        else:
            assert 0
        x = self.quantize(x, ir, 'input')
        opcode = ir['type']
        if opcode == 1:
            in_channels = ir['input_size'][-1]
            out_channels  = ir['output_size'][-1]
            kernel_size = tuple(ir['ker_size'][1:3])
            stride = tuple(ir['ker_stride'][1:3])
            groups = ir['group']
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride,
                          groups=groups, bias=False)
            print(conv2d)
            # hwio -> oihw
            weight = torch.from_numpy(params['weight'].transpose(3,2,0,1))
            conv2d.weight = nn.Parameter(weight)
            x = conv2d(x)
            bias = params['bias']
            bias = np.expand_dims(bias, bias.ndim)
            bias = np.expand_dims(bias, bias.ndim)
            bias = torch.from_numpy(bias)
            x = x + bias
        elif opcode == 0:
            _,h,w,c = ir['input_size']
            ci = np.prod(ir['input_size'])
            co = ir['output_size'][-1]
            x = x.reshape(c,h,w).permute(1,2,0)
            
            #x = self.fc_hw(x.detach().numpy(), params['weight'].T, params['bias'])
            #x = torch.from_numpy(x)
            
            #x = x.view(-1, ci)
            #import ipdb;ipdb.set_trace()
            x = x.reshape(1, ci)
            weight = torch.from_numpy(params['weight'])
            fc = nn.Linear(ci, co)
            fc.weight = nn.Parameter(weight)
            bias = torch.from_numpy(params['bias'])
            fc.bias = nn.Parameter(bias)
            print(fc)
            x = fc(x)
        else:
            assert 0
        x = self.pooling(x, ir)
        x = self.activation(x, ir)
        print('output_shape:', x.shape)
        x = self.quantize(x, ir, 'output')
        self.var_dict[int(ir['index'])] = x
        np.save('golden_layer_'+str(int(ir['index']))+'_out.npy', x.detach().numpy())
        print()
        #if int(ir['index']) == 5:
        #    tmp = np.load('tmp.npy').astype(np.float32)
        #    self.var_dict[ir['index']] = torch.from_numpy(tmp)
    
    def float2fix(self, value, frac_len=0, word_len=8, round_method='floor'):
        min_value = -2 ** (word_len - 1)
        max_value = 2 ** (word_len - 1) - 1

        if round_method == 'round':
            fix_value = np.floor(value * (2 ** frac_len) + 0.5)
        else:
            fix_value = np.floor(value * (2 ** frac_len))

        fix_value[fix_value < min_value] = min_value
        fix_value[fix_value > max_value] = max_value
        fix_value = fix_value / (2 ** frac_len)

        return fix_value
    
    def fc_hw(self, ifm, ker, bias, cut=True):
        blk_in_size = 116
        h,w,c = ifm.shape
        ci, co = ker.shape
        out_blk = np.zeros(co)
        ifm = ifm.reshape(h*w, c)
        assert h*w*c == ci
        ker = ker.reshape(h*w, c, co)

        for i in range(0, h*w, blk_in_size):
            fm_blk_ci = ifm[i:i+blk_in_size,:]
            ker_blk_ci = ker[i:i+blk_in_size,:,:]
            for j in range(0, c, 64):
                fm_blk = fm_blk_ci[:,j:j+64].reshape(fm_blk_ci.shape[0]*64)
                ker_blk = ker_blk_ci[:,j:j+64,:].reshape((fm_blk_ci.shape[0]*64, co))
                print(fm_blk.shape, ker_blk.shape)
                tmp = fm_blk.dot(ker_blk)
                #import ipdb;ipdb.set_trace()
                if i == 0 and j == 0:
                    out_blk += bias
                out_blk += tmp
                if cut:
                    out_blk = self.float2fix(out_blk, frac_len=11, word_len=16, round_method='floor')
                import ipdb;ipdb.set_trace()
                print()
        if cut:
            out_blk = self.float2fix(out_blk, frac_len=5, word_len=8, round_method='round')
        return out_blk
    
        
    def run(self):
        layer_num = len(self.ir)
        for i in range(layer_num):
            print('LAYER', i)
            params = {}
            # collect inputs
            params['input'] = [self.var_dict[x] for x in self.ir[i+1]['input_layer']]
            path_w = os.path.join(self.wdir, 'weights_'+str(i)+'.npy')
            params['weight'] = np.load(path_w)
            path_b = os.path.join(self.wdir, 'bias_'+str(i)+'.npy')
            params['bias'] = np.load(path_b)
            # run
            self.run_layer(self.ir[i + 1], params)
   
args = parser.parse_args()           
GoldenGen(args).run()   
