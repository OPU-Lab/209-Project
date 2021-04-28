import pytest
import os, shutil
import numpy as np
import subprocess
import logging

logging.basicConfig(filename="compile.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)   

def run_shell_cmd(args):
    cmd = ' '.join(args).strip()
    # logger.info(cmd)
    # subprocess.call(cmd, shell=True)
    p = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print('Error from :', cmd)
        print(p.stderr)
        assert 0

def test_tiny_yolo():
    cwd = os.getcwd()
    # create separate folder
    test_name = 'tinyyolo'
    if not os.path.exists(test_name):
        os.makedirs(test_name)
        run_shell_cmd(['cp', '../example/'+test_name+'/download.sh', test_name+'/.'])
    # prepare input 
    os.chdir(test_name)
    if not os.path.exists(test_name):
        os.makedirs(test_name)
    os.chdir(test_name)
    if not os.path.exists('weights.bin'):
        run_shell_cmd(['sh', '../download.sh'])
    # config running parameters
    keyword = 'TINYYOLO'
    os.chdir(os.path.join(cwd, '../fsim'))
    if not os.path.exists('config.h.0'):
        run_shell_cmd(['mv', 'config.h', 'config.h.0'])
    wlines = []
    with open('config.h.0') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if '#if' in line and keyword in line:
                j = 0
                while '#endif' not in lines[i+j]:
                    wlines.append(lines[i+j])
                    j = j + 1
                wlines.append(lines[i+j])
                break
                
    with open('config.h', 'w') as fw:
        print('#define DEBUG_POST_OUT', file=fw)
        print('#define ' + keyword, file=fw)
        for line in wlines:
            print(line, end='', file=fw)
    # compile        
    if not os.path.exists('build'):
        os.makedirs('build')
        os.chdir('build')
        run_shell_cmd(['cmake ..'])
        run_shell_cmd(['make'])
    else:
        os.chdir('build')
        run_shell_cmd(['cmake ..'])
        run_shell_cmd(['make'])
    # run flow
    os.chdir(os.path.join(cwd, test_name))
    sim_x = os.path.join(cwd, '../fsim', 'build', 'simulator')
    run_shell_cmd([sim_x])
    # check
    run_shell_cmd(['cp', os.path.join(cwd, '../example', test_name, 'store_output.txt'), './gt.txt'])
    def hex2int(x):
        bstr = bin(int(x, 16))[2:]
        if len(bstr)<26:
            bstr = bstr[0]*(26-len(bstr))+bstr
        elif len(bstr)>26:
            bstr = bstr[-26:]
        return int(bstr,2)
    f = open('gt.txt')
    lines = f.readlines()
    f.close()
    f = open('out.txt')
    lines1 = f.readlines()
    f.close()
    for i in range(len(lines)):
        dat = [hex2int(x) for x in lines[i].strip().split(' ')]
        dat1 = [hex2int(x) for x in lines1[i].strip().split(' ')]
        for j in range(len(dat1)):
            if not dat[j]==dat1[j]:
                print('line',i+1,'(',j+1,')',hex(dat[j]),hex(dat1[j]))
                print(lines[i].strip().split(' ')[j],lines1[i].strip().split(' ')[j])
                os.chdir(os.path.join(cwd, '../fsim'))
                run_shell_cmd(['rm', 'config.h'])
                run_shell_cmd(['mv', 'config.h.0', 'config.h'])
                raise RuntimeError('Unable to match simulation output with ground truth')
                assert 0
    # cleanup
    os.chdir(os.path.join(cwd, '../fsim'))
    run_shell_cmd(['rm', 'config.h'])
    run_shell_cmd(['mv', 'config.h.0', 'config.h'])
    os.chdir(cwd)
    assert 1